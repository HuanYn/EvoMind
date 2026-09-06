import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .norm import RMSNorm
from .rope import apply_rope, precompute_rope_frequencies


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Share each KV head across n_rep query heads (GQA)."""
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=1)


class CausalSelfAttention(nn.Module):
    """MiniMind-main style GQA attention with RoPE, Q/K norm and KV cache."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.dim % config.n_heads or config.n_heads % config.n_kv_heads:
            raise ValueError("dim must divide by n_heads and n_heads by n_kv_heads")

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout
        self.flash = config.flash_attn and hasattr(F, "scaled_dot_product_attention")

        self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.resid_dropout = nn.Dropout(config.dropout)

        cos, sin = precompute_rope_frequencies(
            self.head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, past_key_value=None, use_cache: bool = False):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        past_len = 0 if past_key_value is None else past_key_value[0].size(1)
        positions = torch.arange(past_len, past_len + seq_len, device=x.device)
        if positions[-1] >= self.rope_cos.size(0):
            raise ValueError("sequence exceeds configured max_seq_len")
        q = apply_rope(q, self.rope_cos, self.rope_sin, positions)
        k = apply_rope(k, self.rope_cos, self.rope_sin, positions)

        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=1)
            v = torch.cat((past_key_value[1], v), dim=1)
        present_key_value = (k, v) if use_cache else None

        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)

        # SDPA selects a fused FlashAttention-like kernel when hardware supports it.
        if self.flash and past_len == 0:
            output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            key_positions = torch.arange(k.size(-2), device=x.device)
            query_positions = past_len + torch.arange(seq_len, device=x.device)
            causal_mask = key_positions[None, :] <= query_positions[:, None]
            scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))
            weights = F.softmax(scores.float(), dim=-1).type_as(q)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            output = torch.matmul(weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.resid_dropout(self.o_proj(output)), present_key_value
