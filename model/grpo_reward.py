"""Deterministic reward functions for the first MiniMind GRPO task."""

from __future__ import annotations

import re
from dataclasses import dataclass


_FINAL_INTEGER = re.compile(r"(-?\d+)\s*[。.!！?？]*\s*$")


@dataclass(frozen=True)
class MathReward:
    """Auditable result of exact-integer arithmetic verification."""

    reward: float
    parsed_answer: int | None
    expected_answer: int


def score_final_integer(response: str, expected_answer: int) -> MathReward:
    """Give reward 1 only when the response ends with the correct integer.

    Allowing explanatory text before the final integer makes early rollouts
    usable, while requiring the integer at the end avoids rewarding a number
    merely copied from the question.  This is a deterministic verifier, not an
    LLM judge and not a partial-credit heuristic.
    """
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    if isinstance(expected_answer, bool) or not isinstance(expected_answer, int):
        raise TypeError("expected_answer must be an integer")
    match = _FINAL_INTEGER.search(response)
    parsed = int(match.group(1)) if match else None
    return MathReward(
        reward=1.0 if parsed == expected_answer else 0.0,
        parsed_answer=parsed,
        expected_answer=expected_answer,
    )


__all__ = ["MathReward", "score_final_integer"]
