"""Sparse Mixture-of-Experts feed-forward layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class MoE(nn.Module):
    """Top-k token routing over a set of SwiGLU experts."""

    def __init__(self, dim: int, hidden_dim: int, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        if not 1 <= num_experts_per_tok <= num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(dim, hidden_dim) for _ in range(num_experts)])

    def forward(
        self, x: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return routed output, load-balancing loss, and per-expert token fraction.

        ``token_mask`` excludes padding positions from the router, experts, and
        routing statistics. Excluded positions receive an all-zero MoE output,
        so the surrounding residual connection leaves those positions unchanged.
        """
        batch_size, seq_len, dim = x.shape
        flat_x = x.reshape(-1, dim)

        if token_mask is None:
            valid_indices = None
            routed_x = flat_x
        else:
            if token_mask.dtype is not torch.bool:
                raise TypeError("token_mask must have dtype torch.bool")
            if token_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    "token_mask must have shape [batch_size, seq_len], "
                    f"got {tuple(token_mask.shape)}"
                )
            if token_mask.device != x.device:
                raise ValueError("token_mask and x must be on the same device")

            valid_indices = token_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                raise ValueError("token_mask must select at least one token")
            routed_x = flat_x.index_select(0, valid_indices)

        router_logits = self.gate(routed_x)
        router_probs = F.softmax(router_logits.float(), dim=-1).type_as(routed_x)
        topk_weights, topk_indices = torch.topk(
            router_probs, k=self.num_experts_per_tok, dim=-1
        )

        routed_output = torch.zeros_like(routed_x)
        for expert_index, expert in enumerate(self.experts):
            token_indices, slot_indices = torch.where(topk_indices == expert_index)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(routed_x[token_indices])
            weighted_output = expert_output * topk_weights[token_indices, slot_indices].unsqueeze(-1)
            routed_output.index_add_(0, token_indices, weighted_output)

        if valid_indices is None:
            output = routed_output
        else:
            output = torch.zeros_like(flat_x)
            output.index_copy_(0, valid_indices, routed_output)

        expert_fraction = torch.bincount(
            topk_indices.reshape(-1), minlength=self.num_experts
        ).float() / topk_indices.numel()
        mean_router_probability = router_probs.float().mean(dim=0)
        aux_loss = self.num_experts * torch.sum(expert_fraction * mean_router_probability)
        return output.reshape(batch_size, seq_len, dim), aux_loss, expert_fraction
