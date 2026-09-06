"""GRPO primitives shared by the single-GPU MiniMind reproduction.

The layout deliberately mirrors MiniMind's official ``train_grpo.py``:
rollout-time old log-probabilities, group-relative advantages, token-level
PPO clipping, a frozen-reference KL term, and a completion-only loss mask.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
    """Standardise rewards independently within each prompt's rollout group."""
    if rewards.ndim != 1 or rewards.numel() == 0 or rewards.numel() % group_size:
        raise ValueError("rewards must be a non-empty 1D tensor divisible by group_size")
    if group_size < 2:
        raise ValueError("group_size must be at least 2 for GRPO")
    if eps <= 0:
        raise ValueError("eps must be positive")
    grouped = rewards.reshape(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    return ((grouped - mean) / (std + eps)).reshape_as(rewards)


def completion_logps(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_length: int,
    completion_length: int,
) -> torch.Tensor:
    """Return log p(completion token | all earlier tokens) for every position.

    ``input_ids`` contains ``[prompt, completion]``.  The causal shift means
    the logit at position ``prompt_length - 1 + t`` predicts completion token
    ``t``.  Padding, if present, must be on the right and is removed later by
    the explicit completion mask.
    """
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [N,T,V] and input_ids must be [N,T]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must have the same first two dimensions")
    if prompt_length < 1 or completion_length < 1:
        raise ValueError("prompt_length and completion_length must be positive")
    start = prompt_length - 1
    end = start + completion_length
    if end > logits.size(1) or prompt_length + completion_length > input_ids.size(1):
        raise ValueError("prompt/completion boundaries exceed the sequence")
    token_ids = input_ids[:, prompt_length : prompt_length + completion_length]
    selected_logits = logits[:, start:end, :]
    return torch.log_softmax(selected_logits.float(), dim=-1).gather(
        -1, token_ids.unsqueeze(-1)
    ).squeeze(-1).type_as(logits)


def token_kl(reference_logps: torch.Tensor, policy_logps: torch.Tensor) -> torch.Tensor:
    """Official MiniMind's non-negative sampled per-token KL estimator."""
    delta = reference_logps - policy_logps
    return torch.exp(delta) - delta - 1.0


def grpo_policy_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    epsilon: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute completion-masked clipped GRPO loss plus reference KL.

    This is the standard PPO branch in MiniMind's trainer, expressed as a loss
    to minimise. ``advantages`` is one scalar per generated completion.
    """
    if policy_logps.shape != old_logps.shape or policy_logps.shape != reference_logps.shape:
        raise ValueError("all per-token log-probability tensors must share a shape")
    if policy_logps.ndim != 2:
        raise ValueError("per-token log-probabilities must be [N, completion_length]")
    if advantages.shape != (policy_logps.size(0),):
        raise ValueError("advantages must have one value per completion")
    if completion_mask.shape != policy_logps.shape or completion_mask.dtype is not torch.bool:
        raise ValueError("completion_mask must be bool and match per-token log-probabilities")
    if not 0 <= epsilon < 1 or beta < 0:
        raise ValueError("epsilon must be in [0, 1) and beta must be non-negative")
    if not completion_mask.any():
        raise ValueError("GRPO batch contains no completion tokens")

    ratio = torch.exp(policy_logps - old_logps)
    clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    advantage = advantages.unsqueeze(1)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    kl = token_kl(reference_logps, policy_logps)
    per_token_loss = -(surrogate - beta * kl)
    mask = completion_mask.to(per_token_loss.dtype)
    per_completion = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    loss = per_completion.mean()
    diagnostics = {
        "policy_loss": (-(surrogate * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).mean(),
        "kl": (kl * mask).sum() / mask.sum().clamp_min(1),
        "ratio": (ratio * mask).sum() / mask.sum().clamp_min(1),
        "clip_fraction": (((ratio - clipped_ratio).abs() > 0).to(mask.dtype) * mask).sum() / mask.sum().clamp_min(1),
    }
    return loss, diagnostics


__all__ = ["completion_logps", "group_advantages", "grpo_policy_loss", "token_kl"]
