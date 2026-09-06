"""In-memory JSONL dataset for supervised fine-tuning subsets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .chat_template import IGNORE_INDEX, encode_sft_conversation


# Keep this list aligned with ``dataset/lm_dataset.py`` in the official
# MiniMind repository.  Exposing it at module level also makes the exact data
# augmentation policy inspectable and testable.
SYSTEM_PROMPTS = (
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是minimind，一个小巧但有用的语言模型。",
    "你是一个专业的AI助手，请提供有价值的回答。",
    "你是minimind，请尽力帮助用户解决问题。",
    "你是一个可靠的AI，请给出准确的回答。",
    "You are a helpful AI assistant.",
    "You are minimind, a lightweight intelligent assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are minimind, a small but useful language model.",
)


class SFTDataset(Dataset):
    """Load a curated JSONL subset and create right-padded SFT examples.

    This deliberately keeps the parsed conversations in memory. It is intended
    for the sampled/curated SFT subset prepared by the project, not for loading
    the original multi-gigabyte JSONL corpus in full.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_length: int = 768,
        seed: int = 42,
        empty_think_ratio: float = 0.2,
        system_prompt_ratio: float = 0.2,
    ) -> None:
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
            raise ValueError("max_length must be an integer of at least 2 for shifted causal loss")
        if (
            isinstance(empty_think_ratio, bool)
            or not isinstance(empty_think_ratio, (int, float))
            or not math.isfinite(empty_think_ratio)
            or not 0.0 <= empty_think_ratio <= 1.0
        ):
            raise ValueError("empty_think_ratio must be a finite number in [0, 1]")
        if (
            isinstance(system_prompt_ratio, bool)
            or not isinstance(system_prompt_ratio, (int, float))
            or not math.isfinite(system_prompt_ratio)
            or not 0.0 <= system_prompt_ratio <= 1.0
        ):
            raise ValueError("system_prompt_ratio must be a finite number in [0, 1]")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"SFT JSONL file not found: {self.path}")

        try:
            pad_id = tokenizer.pad_id
        except AttributeError as error:
            raise TypeError("tokenizer must expose an integer pad_id property") from error
        if isinstance(pad_id, bool) or not isinstance(pad_id, int) or pad_id < 0:
            raise ValueError("tokenizer.pad_id must be a non-negative integer")

        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.max_length = max_length
        self.seed = seed
        self.empty_think_ratio = float(empty_think_ratio)
        self.system_prompt_ratio = float(system_prompt_ratio)
        self.conversations: list[list[dict[str, Any]]] = []
        self.line_numbers: list[int] = []

        with self.path.open(encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed JSON in {self.path} at line {line_number}: {error.msg}"
                    ) from error

                conversations = record.get("conversations") if isinstance(record, dict) else None
                if not isinstance(conversations, list) or not conversations:
                    raise ValueError(
                        f"Invalid SFT record in {self.path} at line {line_number}: "
                        "'conversations' must be a non-empty list"
                    )
                if any(not isinstance(message, dict) for message in conversations):
                    raise ValueError(
                        f"Invalid SFT record in {self.path} at line {line_number}: "
                        "every conversation message must be an object"
                    )

                self.conversations.append(conversations)
                self.line_numbers.append(line_number)

        if not self.conversations:
            raise ValueError(f"No non-empty SFT records found in {self.path}")

    def __len__(self) -> int:
        return len(self.conversations)

    def _keep_empty_think(self, index: int) -> bool:
        if self.empty_think_ratio == 0.0:
            return False
        if self.empty_think_ratio == 1.0:
            return True

        # Hashing (seed, index) makes the decision independent of access order,
        # DataLoader worker count, and global Python/PyTorch RNG state.
        key = f"{self.seed}:{index}".encode("utf-8")
        random_bits = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        return random_bits / 2**64 < self.empty_think_ratio

    def _system_prompt_decision(self, index: int) -> bool:
        """Return a reproducible Bernoulli decision for one dataset index."""

        if self.system_prompt_ratio == 0.0:
            return False
        if self.system_prompt_ratio == 1.0:
            return True

        # Use a distinct hash namespace from empty-think sampling so changing
        # one augmentation never changes the other one's decisions.
        key = f"system-prompt:add:{self.seed}:{index}".encode("utf-8")
        random_bits = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        return random_bits / 2**64 < self.system_prompt_ratio

    def _system_prompt(self, index: int) -> str:
        """Choose one official prompt deterministically for one dataset index."""

        key = f"system-prompt:choice:{self.seed}:{index}".encode("utf-8")
        random_bits = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        return SYSTEM_PROMPTS[random_bits % len(SYSTEM_PROMPTS)]

    @staticmethod
    def _contains_tool_use(conversations: list[dict[str, Any]]) -> bool:
        """Detect tool-enabled samples, which official MiniMind leaves intact."""

        return any(
            message.get("tools")
            or message.get("tool_calls")
            or message.get("role") == "tool"
            for message in conversations
        )

    def _conversation_for_index(self, index: int) -> list[dict[str, Any]]:
        """Return the original conversation or an augmented, new outer list."""

        conversations = self.conversations[index]
        if (
            conversations[0].get("role") != "system"
            and not self._contains_tool_use(conversations)
            and self._system_prompt_decision(index)
        ):
            # ``+`` creates a new list.  Neither the source JSON record nor the
            # stored conversation is mutated by this augmentation.
            return [
                {"role": "system", "content": self._system_prompt(index)},
                *conversations,
            ]
        return conversations

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("SFTDataset index out of range")

        source_conversation = self.conversations[index]
        encoded_conversation = self._conversation_for_index(index)
        input_ids, labels = encode_sft_conversation(
            encoded_conversation,
            self.tokenizer,
            max_length=self.max_length,
            keep_empty_think=self._keep_empty_think(index),
        )

        # A synthetic system turn consumes part of the fixed context window.
        # For a very tight max_length it can push every assistant target past
        # the right-truncation boundary.  In that case keep the usable training
        # example by deterministically retrying the unaugmented conversation.
        # The original encoding still goes through the normal validation below.
        has_shifted_target = len(input_ids) >= 2 and any(
            label != IGNORE_INDEX for label in labels[1:]
        )
        if encoded_conversation is not source_conversation and not has_shifted_target:
            input_ids, labels = encode_sft_conversation(
                source_conversation,
                self.tokenizer,
                max_length=self.max_length,
                keep_empty_think=self._keep_empty_think(index),
            )

        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long)

        if input_ids.ndim != 1 or labels.ndim != 1:
            raise ValueError(
                f"Encoded SFT record at line {self.line_numbers[index]} must be one-dimensional"
            )
        if input_ids.numel() != labels.numel():
            raise ValueError(
                f"Encoded SFT record at line {self.line_numbers[index]} has mismatched "
                "input_ids and labels lengths"
            )
        if input_ids.numel() > self.max_length:
            raise ValueError(
                f"Encoded SFT record at line {self.line_numbers[index]} exceeds "
                f"max_length={self.max_length}"
            )
        if input_ids.numel() < 2 or not torch.any(labels[1:] != IGNORE_INDEX):
            raise ValueError(
                f"SFT record at line {self.line_numbers[index]} has no supervised assistant "
                f"target after truncation to max_length={self.max_length}"
            )

        padding_length = self.max_length - input_ids.numel()
        if padding_length:
            input_ids = torch.cat(
                (input_ids, torch.full((padding_length,), self.pad_id, dtype=torch.long))
            )
            labels = torch.cat(
                (labels, torch.full((padding_length,), IGNORE_INDEX, dtype=torch.long))
            )

        return input_ids, labels


__all__ = ["SYSTEM_PROMPTS", "SFTDataset"]
