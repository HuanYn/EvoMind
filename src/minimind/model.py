import torch
import torch.nn as nn

from .block import TransformerBlock
from .config import ModelConfig


class MiniMindModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.dim,
        )
        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.dim,
        )
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError("input sequence is longer than max_seq_len")

        position_ids = torch.arange(
            seq_len,
            device=input_ids.device,
        )

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(position_ids)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits