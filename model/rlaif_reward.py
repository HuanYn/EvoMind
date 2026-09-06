"""MiniMind-style composite rewards for RLAIF GRPO rollouts.

The public RLAIF data only supplies a prompt/history.  A reward is computed
*after* the policy has generated a continuation: inexpensive format controls
are added to a frozen preference reward-model score.  Keeping this module
model-agnostic makes every non-neural term unit-testable and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RLAIFReward:
    """Breakdown of one generated answer's MiniMind-style reward."""

    reward: float
    length_reward: float
    think_reward: float
    repetition_penalty: float
    reward_model_score: float


def clipped_reward_model_score(score: float, bound: float = 3.0) -> float:
    """Bound a scalar RM score so an outlier cannot dominate a rollout group."""

    if bound <= 0:
        raise ValueError("bound must be positive")
    return max(-bound, min(bound, float(score)))


def repeated_ngram_penalty(text: str, ngram_size: int = 3, cap: float = 0.5) -> float:
    """Return a bounded repetition penalty using character n-grams.

    Chinese has no reliable whitespace word boundary, so the local
    reproduction measures repeated *character* spans.  The penalty is the
    fraction of duplicate n-grams, capped at ``cap``; zero means no repeated
    n-gram occurred.
    """

    if ngram_size < 1 or cap < 0:
        raise ValueError("ngram_size must be positive and cap must be non-negative")
    compact = "".join(text.split())
    if len(compact) < ngram_size:
        return 0.0
    grams = [compact[index : index + ngram_size] for index in range(len(compact) - ngram_size + 1)]
    duplicate_fraction = 1.0 - len(set(grams)) / len(grams)
    return min(cap, duplicate_fraction)


def score_minimind_rlaif(
    response: str,
    reward_model_score: float,
    *,
    min_response_chars: int = 20,
    max_response_chars: int = 800,
    min_think_chars: int = 20,
    max_think_chars: int = 300,
) -> RLAIFReward:
    """Score one rollout by MiniMind's documented reward components.

    The rule weights follow the official training recipe: a bounded response
    gets +/-0.5; a present reasoning block receives a length score (+1/-0.5)
    and a closure-format score (+0.25/-0.25); repeated text is penalised by up
    to 0.5; finally a frozen RM score is clipped to [-3, 3] and added.
    """

    if not isinstance(response, str):
        raise TypeError("response must be a string")
    if not (0 <= min_response_chars <= max_response_chars and 0 <= min_think_chars <= max_think_chars):
        raise ValueError("invalid length bounds")

    response_length = len(response.strip())
    length_reward = 0.5 if min_response_chars <= response_length <= max_response_chars else -0.5

    think_reward = 0.0
    if "</think>" in response:
        think_end = response.find("</think>")
        think_start = response.rfind("<think>", 0, think_end)
        reasoning = response[think_start + len("<think>") : think_end] if think_start >= 0 else response[:think_end]
        think_reward += 1.0 if min_think_chars <= len(reasoning.strip()) <= max_think_chars else -0.5
        think_reward += 0.25 if response.count("</think>") == 1 else -0.25

    repetition_penalty = repeated_ngram_penalty(response)
    bounded_rm = clipped_reward_model_score(reward_model_score)
    total = length_reward + think_reward - repetition_penalty + bounded_rm
    return RLAIFReward(total, length_reward, think_reward, repetition_penalty, bounded_rm)


__all__ = ["RLAIFReward", "clipped_reward_model_score", "repeated_ngram_penalty", "score_minimind_rlaif"]
