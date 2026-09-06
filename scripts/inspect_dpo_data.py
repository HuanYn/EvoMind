"""Audit preference-pair JSONL before using it for DPO training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from uuid import uuid4

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.chat_template import encode_sft_conversation  # noqa: E402
from model.tokenizer import BPETokenizer  # noqa: E402


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_profile(messages: list[dict[str, str]]) -> tuple[int, int, int]:
    text = "".join(message["content"] for message in messages)
    chinese = sum("\u4e00" <= char <= "\u9fff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    return len(text), chinese, latin


def validate_conversation(value: object, field: str, line_number: int) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"line {line_number}: {field} must contain at least user and assistant")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"line {line_number}: {field}[{index}] is not an object")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
            raise ValueError(f"line {line_number}: invalid {field}[{index}] role/content")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"line {line_number}: {field} must end with assistant")
    return messages


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit chosen/rejected preference pairs for MiniMind DPO")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file() or not args.tokenizer.is_file():
        raise FileNotFoundError("--input and --tokenizer must exist")
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")

    tokenizer = BPETokenizer.load(args.tokenizer)
    seen: set[str] = set()
    total = duplicates = same_prefix_failures = same_response_pairs = 0
    roles: Counter[str] = Counter()
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    pair_language_buckets: Counter[str] = Counter()
    chosen_chars = chosen_chinese = chosen_latin = 0
    rejected_chars = rejected_chinese = rejected_latin = 0

    with args.input.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                pair = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON") from error
            if not isinstance(pair, dict) or set(pair) != {"chosen", "rejected"}:
                raise ValueError(f"line {line_number}: expected exactly chosen and rejected fields")
            chosen = validate_conversation(pair["chosen"], "chosen", line_number)
            rejected = validate_conversation(pair["rejected"], "rejected", line_number)
            pair_key = canonical_json(pair)
            if pair_key in seen:
                duplicates += 1
                continue
            seen.add(pair_key)
            total += 1
            roles.update(message["role"] for message in chosen + rejected)
            if chosen[:-1] != rejected[:-1]:
                same_prefix_failures += 1
            if chosen[-1]["content"] == rejected[-1]["content"]:
                same_response_pairs += 1

            chosen_ids, _ = encode_sft_conversation(chosen, tokenizer, max_length=args.max_length)
            rejected_ids, _ = encode_sft_conversation(rejected, tokenizer, max_length=args.max_length)
            chosen_lengths.append(len(chosen_ids))
            rejected_lengths.append(len(rejected_ids))
            chars, chinese, latin = text_profile(chosen)
            chosen_chars += chars
            chosen_chinese += chinese
            chosen_latin += latin
            rejected_text_chars, rejected_text_chinese, rejected_text_latin = text_profile(rejected)
            rejected_chars += rejected_text_chars
            rejected_chinese += rejected_text_chinese
            rejected_latin += rejected_text_latin
            pair_chars = chars + rejected_text_chars
            pair_chinese = chinese + rejected_text_chinese
            pair_latin = latin + rejected_text_latin
            if pair_chinese / max(pair_chars, 1) >= 0.5:
                pair_language_buckets["Chinese_dominant"] += 1
            elif pair_latin / max(pair_chars, 1) >= 0.5:
                pair_language_buckets["ASCII_latin_dominant"] += 1
            else:
                pair_language_buckets["mixed_or_other"] += 1

    if total == 0:
        raise ValueError("no non-empty DPO pairs found")
    report = {
        "schema_version": 1,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "max_length": args.max_length,
        "unique_pairs": total,
        "exact_duplicate_pairs_removed": duplicates,
        "prefix_mismatches": same_prefix_failures,
        "identical_chosen_rejected_answers": same_response_pairs,
        "roles": dict(sorted(roles.items())),
        "pair_language_buckets": dict(sorted(pair_language_buckets.items())),
        "chosen": {
            "p50_tokens": percentile(chosen_lengths, 0.50),
            "p95_tokens": percentile(chosen_lengths, 0.95),
            "truncated_at_max_length": sum(length >= args.max_length for length in chosen_lengths),
            "chinese_character_ratio": chosen_chinese / max(chosen_chars, 1),
            "ascii_letter_ratio": chosen_latin / max(chosen_chars, 1),
        },
        "rejected": {
            "p50_tokens": percentile(rejected_lengths, 0.50),
            "p95_tokens": percentile(rejected_lengths, 0.95),
            "truncated_at_max_length": sum(length >= args.max_length for length in rejected_lengths),
            "chinese_character_ratio": rejected_chinese / max(rejected_chars, 1),
            "ascii_letter_ratio": rejected_latin / max(rejected_chars, 1),
        },
    }
    atomic_write(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"unique pairs: {total:,} | exact duplicate pairs removed: {duplicates:,}")
    print(f"prefix mismatches: {same_prefix_failures:,} | identical answers: {same_response_pairs:,}")
    for name in ("chosen", "rejected"):
        stats = report[name]
        print(
            f"{name}: p50={stats['p50_tokens']} | p95={stats['p95_tokens']} | "
            f"truncated={stats['truncated_at_max_length']:,} | Chinese={stats['chinese_character_ratio']:.2%}"
        )
    print(f"saved report: {args.report}")


if __name__ == "__main__":
    main()
