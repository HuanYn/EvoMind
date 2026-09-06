import torch
import torch.nn as nn

from .block import TransformerBlock
from .config import ModelConfig
from .norm import RMSNorm


class MiniMindModel(nn.Module):
    """Decoder-only MiniMind dense model, aligned with the official main design."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Use a small LLM-style normal initialization before tying embeddings."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
        return_router_loss: bool = False,
        token_mask: torch.Tensor | None = None,
    ):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("input sequence is longer than max_seq_len")
        if token_mask is not None:
            if token_mask.dtype is not torch.bool:
                raise TypeError("token_mask must have dtype torch.bool")
            if token_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    "token_mask must have shape [batch_size, seq_len], "
                    f"got {tuple(token_mask.shape)}"
                )
            if token_mask.device != input_ids.device:
                raise ValueError("token_mask and input_ids must be on the same device")

        past_key_values = past_key_values or [None] * len(self.blocks)
        if len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one entry per Transformer block")

        x = self.dropout(self.token_embedding(input_ids))
        next_key_values = []
        router_aux_loss = x.new_zeros(())
        expert_fractions = []
        for block, past_key_value in zip(self.blocks, past_key_values):
            x, present_key_value, block_aux_loss, expert_fraction = block(
                x,
                past_key_value=past_key_value,
                use_cache=use_cache,
                token_mask=token_mask,
            )
            router_aux_loss = router_aux_loss + block_aux_loss
            if expert_fraction is not None:
                expert_fractions.append(expert_fraction)
            if use_cache:
                next_key_values.append(present_key_value)

        logits = self.lm_head(self.final_norm(x))
        if use_cache:
            return (logits, next_key_values) if not return_router_loss else (
                logits, next_key_values, router_aux_loss, expert_fractions
            )
        return logits if not return_router_loss else (logits, router_aux_loss, expert_fractions)
