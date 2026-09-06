"""Preference-pair dataset used by Direct Preference Optimization (DPO)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .chat_template import IGNORE_INDEX, encode_sft_conversation


class DPODataset(Dataset):
    """Load clean ``chosen``/``rejected`` conversation pairs for DPO.

    Each item contains two independently completed conversations with the same
    prompt prefix.  We intentionally do *not* apply SFT's system-prompt or
    empty-think augmentation here: DPO must compare exactly the supplied
    preferred and dispreferred answers under the same condition.
    """

    def __init__(self, path: str | Path, tokenizer: Any, max_length: int = 768) -> None:
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
            raise ValueError("max_length must be an integer of at least 2 for shifted causal loss")

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"DPO JSONL file not found: {self.path}")
        try:
            pad_id = tokenizer.pad_id
        except AttributeError as error:
            raise TypeError("tokenizer must expose an integer pad_id property") from error
        if isinstance(pad_id, bool) or not isinstance(pad_id, int) or pad_id < 0:
            raise ValueError("tokenizer.pad_id must be a non-negative integer")

        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.max_length = max_length
        self.pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        self.line_numbers: list[int] = []

        with self.path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed JSON in {self.path} at line {line_number}: {error.msg}"
                    ) from error
                chosen, rejected = self._validate_pair(record, line_number)
                self.pairs.append((chosen, rejected))
                self.line_numbers.append(line_number)

        if not self.pairs:
            raise ValueError(f"No non-empty DPO pairs found in {self.path}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _validate_pair(
        self, record: object, line_number: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not isinstance(record, dict) or set(record) != {"chosen", "rejected"}:
            raise ValueError(
                f"Invalid DPO record in {self.path} at line {line_number}: "
                "expected exactly 'chosen' and 'rejected'"
            )
        chosen = self._validate_conversation(record["chosen"], line_number, "chosen")
        rejected = self._validate_conversation(record["rejected"], line_number, "rejected")
        if chosen[:-1] != rejected[:-1]:
            raise ValueError(
                f"DPO record in {self.path} at line {line_number} has mismatched prompt prefixes"
            )
        if chosen[-1].get("content") == rejected[-1].get("content"):
            raise ValueError(
                f"DPO record in {self.path} at line {line_number} has identical answers"
            )
        return chosen, rejected

    def _validate_conversation(
        self, value: object, line_number: int, branch: str
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list) or len(value) < 2 or any(
            not isinstance(message, dict) for message in value
        ):
            raise ValueError(
                f"Invalid {branch} conversation in {self.path} at line {line_number}"
            )
        if value[-1].get("role") != "assistant" or not isinstance(value[-1].get("content"), str):
            raise ValueError(
                f"DPO {branch} conversation in {self.path} at line {line_number} "
                "must end in an assistant text message"
            )
        return value

    def _encode_branch(
        self, conversations: list[dict[str, Any]], line_number: int, branch: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = encode_sft_conversation(
            conversations,
            self.tokenizer,
            # Request one additional token so that a naturally exact-length
            # sequence remains legal while a would-be truncation is visible.
            max_length=self.max_length + 1,
            keep_empty_think=False,
        )
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long)
        if input_ids.numel() != labels.numel() or input_ids.numel() < 2:
            raise ValueError(f"DPO {branch} record at line {line_number} has invalid encoding")
        if input_ids.numel() > self.max_length:
            raise ValueError(
                f"DPO {branch} record at line {line_number} reaches max_length={self.max_length}; "
                "prepare DPO data by filtering overlength pairs rather than truncating them"
            )
        if not torch.any(labels[1:] != IGNORE_INDEX):
            raise ValueError(
                f"DPO {branch} record at line {line_number} has no supervised assistant target"
            )
        padding = self.max_length - input_ids.numel()
        input_ids = torch.cat((input_ids, torch.full((padding,), self.pad_id, dtype=torch.long)))
        labels = torch.cat((labels, torch.full((padding,), IGNORE_INDEX, dtype=torch.long)))
        return input_ids, labels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("DPODataset index out of range")
        chosen, rejected = self.pairs[index]
        line_number = self.line_numbers[index]
        chosen_ids, chosen_labels = self._encode_branch(chosen, line_number, "chosen")
        rejected_ids, rejected_labels = self._encode_branch(rejected, line_number, "rejected")
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }


__all__ = ["DPODataset"]
