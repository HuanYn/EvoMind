"""Prepare deterministic, length-safe Chinese preference pairs for MiniMind DPO."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_priority(seed: int, namespace: str, canonical: str) -> bytes:
    return hashlib.sha256(f"{seed}:{namespace}:".encode("ascii") + canonical.encode("utf-8")).digest()


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


def valid_conversation(value: object) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    messages: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, dict):
            return None
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
            return None
        messages.append({"role": role, "content": content})
    return messages if messages[-1]["role"] == "assistant" else None


def chinese_ratio(chosen: list[dict[str, str]], rejected: list[dict[str, str]]) -> float:
    text = "".join(message["content"] for message in chosen + rejected)
    return sum("\u4e00" <= char <= "\u9fff" for char in text) / max(len(text), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare length-safe Chinese DPO pairs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--min-chinese-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.input.is_file() or not args.tokenizer.is_file():
        raise FileNotFoundError("--input and --tokenizer must exist")
    if min(args.train_size, args.val_size, args.max_length) < 1 or not 0 <= args.min_chinese_ratio <= 1:
        raise ValueError("invalid sizes, max-length, or min-chinese-ratio")
    if len({path.resolve() for path in (args.input, args.train_output, args.val_output, args.report)}) != 4:
        raise ValueError("input, train-output, val-output, and report must differ")

    tokenizer = BPETokenizer.load(args.tokenizer)
    seen: set[str] = set()
    candidates: list[str] = []
    rejected = Counter()
    with args.input.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            try:
                pair = json.loads(raw_line)
            except json.JSONDecodeError:
                rejected["invalid_json"] += 1
                continue
            if not isinstance(pair, dict) or set(pair) != {"chosen", "rejected"}:
                rejected["invalid_schema"] += 1
                continue
            chosen, loser = valid_conversation(pair["chosen"]), valid_conversation(pair["rejected"])
            if chosen is None or loser is None:
                rejected["invalid_conversation"] += 1
                continue
            canonical = canonical_json({"chosen": chosen, "rejected": loser})
            if canonical in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(canonical)
            if chosen[:-1] != loser[:-1]:
                rejected["prefix_mismatch"] += 1
            elif chosen[-1]["content"] == loser[-1]["content"]:
                rejected["identical_answers"] += 1
            elif chinese_ratio(chosen, loser) < args.min_chinese_ratio:
                rejected["language"] += 1
            else:
                chosen_ids, _ = encode_sft_conversation(chosen, tokenizer, max_length=32_768)
                loser_ids, _ = encode_sft_conversation(loser, tokenizer, max_length=32_768)
                if len(chosen_ids) > args.max_length or len(loser_ids) > args.max_length:
                    rejected["over_length"] += 1
                else:
                    candidates.append(canonical)
    required = args.train_size + args.val_size
    if len(candidates) < required:
        raise ValueError(f"only {len(candidates)} clean pairs, need {required}")
    ordered = sorted(candidates, key=lambda item: stable_priority(args.seed, "dpo-sample", item))
    val_records = ordered[: args.val_size]
    train_records = ordered[args.val_size : required]
    atomic_write(args.train_output, "".join(record + "\n" for record in train_records))
    atomic_write(args.val_output, "".join(record + "\n" for record in val_records))
    report = {
        "schema_version": 1,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "max_length": args.max_length,
        "min_chinese_ratio": args.min_chinese_ratio,
        "seed": args.seed,
        "clean_candidates": len(candidates),
        "rejected": dict(sorted(rejected.items())),
        "train": {"path": str(args.train_output), "pairs": len(train_records), "sha256": sha256_file(args.train_output)},
        "val": {"path": str(args.val_output), "pairs": len(val_records), "sha256": sha256_file(args.val_output)},
    }
    atomic_write(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"clean candidates: {len(candidates):,} | rejected: {dict(sorted(rejected.items()))}")
    print(f"train: {len(train_records):,} -> {args.train_output}")
    print(f"val: {len(val_records):,} -> {args.val_output}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
