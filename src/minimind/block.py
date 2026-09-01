import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .config import ModelConfig
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attention_norm = nn.LayerNorm(config.dim)
        self.attention = CausalSelfAttention(config)

        self.mlp_norm = nn.LayerNorm(config.dim)
        self.mlp = MLP(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x