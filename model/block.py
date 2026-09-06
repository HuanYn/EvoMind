import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .config import ModelConfig
from .mlp import MLP
from .moe import MoE
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-Norm decoder block following MiniMind main's dense path."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.dim, config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.dim, config.rms_norm_eps)
        self.mlp = (
            MoE(config.dim, config.hidden_dim, config.num_experts, config.num_experts_per_tok)
            if config.use_moe
            else MLP(config.dim, config.hidden_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value=None,
        use_cache: bool = False,
        token_mask: torch.Tensor | None = None,
    ):
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(x), past_key_value=past_key_value, use_cache=use_cache
        )
        x = x + attn_output
        mlp_input = self.post_attention_layernorm(x)
        if isinstance(self.mlp, MoE):
            mlp_output, router_aux_loss, expert_fraction = self.mlp(
                mlp_input, token_mask=token_mask
            )
        else:
            mlp_output = self.mlp(mlp_input)
            router_aux_loss = x.new_zeros(())
            expert_fraction = None
        x = x + mlp_output
        return x, present_key_value, router_aux_loss, expert_fraction
