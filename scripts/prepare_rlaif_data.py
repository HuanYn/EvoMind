"""Prepare the official MiniMind RLAIF prompt pool for local GRPO.

Each raw record ends in an intentionally empty assistant turn.  This script
removes only that placeholder, preserves the preceding multi-turn history,
deduplicates exact histories, creates deterministic disjoint subsets, and
writes an audit report with source/tokenizer hashes and context statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.chat_template import TEMPLATE_VERSION, encode_generation_prompt
from model.tokenizer import BPETokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def priority(seed: int, canonical: str) -> tuple[int, str]:
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    value = int.from_bytes(
        hashlib.blake2b(f"{seed}:rlaif:".encode("ascii") + fingerprint.encode("ascii"), digest_size=8).digest(),
        "big",
    )
    return value, fingerprint


def validate_record(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("conversations"), list):
        raise ValueError("record must contain conversations list")
    conversations = raw["conversations"]
    if len(conversations) < 2:
        raise ValueError("RLAIF conversation must include history and empty assistant placeholder")
    for index, message in enumerate(conversations):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise ValueError(f"message {index} must have string role and content")
    final = conversations[-1]
    if final["role"] != "assistant" or final["content"].strip():
        raise ValueError("final RLAIF turn must be an empty assistant placeholder")
    history = conversations[:-1]
    if history[-1]["role"] != "user":
        raise ValueError("RLAIF rollout history must end with a user message")
    return history


def summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    return {
        "mean": sum(values) / len(values),
        "p50": ordered[(len(ordered) - 1) // 2],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official MiniMind RLAIF prompt-only GRPO data")
    parser.add_argument("--input", type=Path, default=Path("data/raw/minimind_rlaif/rlaif.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/minimind_bpe_16k_110k.json"))
    parser.add_argument("--train-output", type=Path, default=Path("data/processed/minimind_rlaif/rlaif_train_15k_seed42.jsonl"))
    parser.add_argument("--val-output", type=Path, default=Path("data/processed/minimind_rlaif/rlaif_val_500_seed42.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/grpo/rlaif_prepare_15k_500_seed42.json"))
    parser.add_argument("--train-size", type=int, default=15_000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.train_size < 1 or args.val_size < 1 or args.max_length < 1:
        raise ValueError("sizes and max-length must be positive")
    if not args.input.is_file() or not args.tokenizer.is_file():
        raise FileNotFoundError("input and tokenizer must exist")
    for output in (args.train_output, args.val_output, args.report):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    tokenizer = BPETokenizer.load(args.tokenizer)
    unique: dict[str, tuple[int, str, list[dict[str, object]]]] = {}
    invalid = duplicates = nonempty = 0
    raw_prompt_lengths: list[int] = []
    truncated = 0
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        nonempty += 1
        try:
            history = validate_record(json.loads(line))
            canonical = json.dumps({"messages": history}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            rank, fingerprint = priority(args.seed, canonical)
            if fingerprint in unique:
                duplicates += 1
                continue
            full = encode_generation_prompt(history, tokenizer, max_length=2**31 - 1)
            _ = encode_generation_prompt(history, tokenizer, max_length=args.max_length)
            raw_prompt_lengths.append(len(full))
            truncated += int(len(full) > args.max_length)
            unique[fingerprint] = (rank, canonical, history)
        except (TypeError, ValueError, json.JSONDecodeError):
            invalid += 1

    ordered = sorted(unique.values(), key=lambda item: (item[0], item[1]))
    required = args.train_size + args.val_size
    if len(ordered) < required:
        raise ValueError(f"only {len(ordered)} valid unique prompts, but {required} requested")
    selected = ordered[:required]
    val, train = selected[: args.val_size], selected[args.val_size :]

    def write(path: Path, rows: list[tuple[int, str, list[dict[str, object]]]], split: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for index, (_, canonical, _) in enumerate(rows):
                record = json.loads(canonical)
                record["id"] = f"rlaif-{split}-{index:06d}"
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    write(args.train_output, train, "train")
    write(args.val_output, val, "val")
    report = {
        "stage": "grpo_rlaif_prepare",
        "source": {"path": str(args.input), "sha256": sha256_file(args.input), "nonempty_records": nonempty},
        "tokenizer": {"path": str(args.tokenizer), "sha256": sha256_file(args.tokenizer), "vocab_size": tokenizer.vocab_size},
        "template_version": TEMPLATE_VERSION,
        "seed": args.seed,
        "raw_contract": "conversations ending in an empty assistant placeholder; placeholder is removed for rollout",
        "unique_records": len(unique),
        "exact_duplicates_removed": duplicates,
        "invalid_records": invalid,
        "prompt_tokens_before_left_truncation": summary(raw_prompt_lengths),
        "prompts_exceeding_max_length": truncated,
        "max_length": args.max_length,
        "train": {"path": str(args.train_output), "records": len(train), "sha256": sha256_file(args.train_output)},
        "validation": {"path": str(args.val_output), "records": len(val), "sha256": sha256_file(args.val_output)},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"source records: {nonempty:,} | unique: {len(unique):,} | duplicates: {duplicates:,} | invalid: {invalid:,}")
    print(f"prompt tokens before left truncation: {report['prompt_tokens_before_left_truncation']}")
    print(f"prompts over {args.max_length}: {truncated:,}")
    print(f"train: {len(train):,} -> {args.train_output} | val: {len(val):,} -> {args.val_output}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
