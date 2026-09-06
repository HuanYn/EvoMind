"""Small, testable math helpers for Direct Preference Optimization."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .chat_template import IGNORE_INDEX


def assistant_sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum shifted log probabilities over assistant tokens for every sample.

    ``logits`` is produced from ``input_ids[:, :-1]`` and hence aligns with
    ``labels[:, 1:]``.  Labels equal to ``IGNORE_INDEX`` are prompt/header/pad
    locations and are excluded from the sum.
    """
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels[:, 1:].shape:
        raise ValueError("expected logits [B, T-1, V] and labels [B, T]")
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every DPO branch must contain at least one assistant target")
    safe_targets = targets.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits.float(), dim=-1).gather(
        dim=-1, index=safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return mean DPO loss and detached-friendly per-example diagnostics."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    tensors = (
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    if any(value.ndim != 1 for value in tensors) or len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("all DPO log-prob tensors must have the same [B] shape")
    chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - reference_rejected_logps)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin).mean()
    return loss, {
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
        "margin": margin,
        "preference_accuracy": margin.gt(0).float(),
    }


__all__ = ["assistant_sequence_logps", "dpo_loss"]
