import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, value)
    return output, weights


def repeat_kv(x, n_rep):
    """让一组 KV 头共享给 n_rep 个 Query 头。"""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.dim,
            self.n_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            config.dim,
            bias=False,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size, seq_len, self.n_heads, self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim,
        ).transpose(1, 2)

        n_rep = self.n_heads // self.n_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        causal_mask = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=x.device,
            dtype=x.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)

        output, _ = attention(
            q,
            k,
            v,
            mask=causal_mask,
            dropout=self.attn_dropout,
        )

        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.n_heads * self.head_dim,
        )

        output = self.o_proj(output)
        return self.resid_dropout(output)