"""Prepare deterministic, leakage-free MiniMind SFT JSONL subsets."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.chat_template import (  # noqa: E402
    IGNORE_INDEX,
    TEMPLATE_VERSION,
    encode_sft_conversation,
)
from model.tokenizer import BPETokenizer  # noqa: E402

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
ALLOWED_MESSAGE_FIELDS = {
    "role",
    "content",
    "reasoning_content",
    "tools",
    "tool_calls",
}
FULL_LENGTH_LIMIT = 2**31 - 1


def stable_u64(seed: int, namespace: str, fingerprint: bytes) -> int:
    """Return a reproducible pseudo-random 64-bit number."""

    payload = f"{seed}:{namespace}:".encode("ascii") + fingerprint
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def canonical_record(record: Any) -> str:
    """Validate one record and canonicalise the conversation used for training."""

    if not isinstance(record, dict):
        raise ValueError("record must be a JSON object")
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations must be a non-empty list")

    tools_messages = 0
    has_assistant_payload = False
    for index, message in enumerate(conversations):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be a JSON object")
        unknown_fields = set(message) - ALLOWED_MESSAGE_FIELDS
        if unknown_fields:
            names = ", ".join(sorted(map(str, unknown_fields)))
            raise ValueError(f"message {index} has unsupported fields: {names}")

        role = message.get("role")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"message {index} has invalid role: {role!r}")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"message {index}.content must be a string")

        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError(f"message {index}.reasoning_content must be text or null")
        if role != "assistant" and reasoning not in (None, ""):
            raise ValueError("reasoning_content is only valid on assistant messages")

        tools = message.get("tools")
        if tools not in (None, ""):
            if role != "system":
                raise ValueError("tools is only valid on system messages")
            tools_messages += 1
            if tools_messages > 1:
                raise ValueError("only one system message may define tools")

        tool_calls = message.get("tool_calls")
        if tool_calls not in (None, "") and role != "assistant":
            raise ValueError("tool_calls is only valid on assistant messages")

        if role == "assistant" and (
            bool(message["content"].strip())
            or isinstance(reasoning, str)
            and bool(reasoning.strip())
            or tool_calls not in (None, "", [], {})
        ):
            has_assistant_payload = True

    if not has_assistant_payload:
        raise ValueError("conversation has no assistant payload to supervise")

    return json.dumps(
        {"conversations": conversations},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _offer_candidate(
    heap: list[tuple[int, int, bytes, str]],
    limit: int,
    priority: int,
    fingerprint: bytes,
    canonical: str,
) -> None:
    """Keep the records with the smallest stable hash priorities."""

    fingerprint_int = int.from_bytes(fingerprint, "big")
    item = (-priority, -fingerprint_int, fingerprint, canonical)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _ordered_candidates(
    heap: list[tuple[int, int, bytes, str]],
) -> list[tuple[int, bytes, str]]:
    candidates = [(-neg_priority, fingerprint, canonical) for neg_priority, _, fingerprint, canonical in heap]
    return sorted(candidates, key=lambda item: (item[0], item[1]))


def _keep_by_ratio(seed: int, index: int, ratio: float) -> bool:
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    key = f"{seed}:{index}".encode("utf-8")
    bits = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return bits / 2**64 < ratio


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


def _summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "p50": 0, "p95": 0, "max": 0}
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _validate_candidates(
    candidates: list[tuple[int, bytes, str]],
    requested: int,
    tokenizer: Any,
    max_length: int,
    seed: int,
    empty_think_ratio: float,
) -> tuple[list[str], dict[str, Any]]:
    selected: list[str] = []
    full_lengths: list[int] = []
    supervised_lengths: list[int] = []
    failures: Counter[str] = Counter()
    truncated = 0

    for _, _, canonical in candidates:
        record = json.loads(canonical)
        keep_empty_think = _keep_by_ratio(seed, len(selected), empty_think_ratio)
        try:
            input_ids, labels = encode_sft_conversation(
                record["conversations"],
                tokenizer,
                max_length=FULL_LENGTH_LIMIT,
                keep_empty_think=keep_empty_think,
            )
        except (TypeError, ValueError) as error:
            failures[type(error).__name__] += 1
            continue

        truncated_labels = labels[:max_length]
        supervised = sum(label != IGNORE_INDEX for label in truncated_labels[1:])
        if supervised == 0:
            failures["no_supervised_target_after_truncation"] += 1
            continue

        full_lengths.append(len(input_ids))
        supervised_lengths.append(supervised)
        truncated += len(input_ids) > max_length
        selected.append(canonical)
        if len(selected) == requested:
            break

    if len(selected) < requested:
        raise ValueError(
            f"Only {len(selected):,} valid records remained for a requested "
            f"{requested:,}; increase --reserve-ratio or inspect source errors"
        )

    return selected, {
        "candidate_records_checked": len(full_lengths) + sum(failures.values()),
        "template_failures": dict(failures),
        "records_over_max_length": truncated,
        "records_over_max_length_rate": truncated / len(selected),
        "full_template_tokens": _summary(full_lengths),
        "supervised_tokens_after_truncation": _summary(supervised_lengths),
    }


def _write_jsonl_atomic(path: Path, records: list[str], overwrite: bool) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            for record in records:
                line = record + "\n"
                target.write(line)
                digest.update(line.encode("utf-8"))
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise
    return digest.hexdigest()


def _write_report_atomic(path: Path, report: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            target.write(json.dumps(report, ensure_ascii=False, indent=2))
            target.write("\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def prepare_sft_subsets(
    *,
    input_path: Path,
    tokenizer: Any,
    train_output: Path,
    val_output: Path,
    smoke_output: Path,
    report_output: Path,
    train_size: int,
    val_size: int,
    smoke_size: int,
    max_length: int,
    val_ratio: float,
    reserve_ratio: float,
    empty_think_ratio: float,
    seed: int,
    tokenizer_path: Path | None = None,
    progress_every: int = 100_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Scan, deduplicate, split, validate and write SFT subsets."""

    if not input_path.is_file():
        raise FileNotFoundError(f"SFT input not found: {input_path}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if train_size < 1 or val_size < 1:
        raise ValueError("train_size and val_size must be positive")
    if smoke_size < 1 or smoke_size > train_size:
        raise ValueError("smoke_size must be in [1, train_size]")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if not math.isfinite(val_ratio) or not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    if not math.isfinite(reserve_ratio) or reserve_ratio < 0.0:
        raise ValueError("reserve_ratio must be non-negative")
    if not math.isfinite(empty_think_ratio) or not 0.0 <= empty_think_ratio <= 1.0:
        raise ValueError("empty_think_ratio must be in [0, 1]")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    resolved_outputs = {
        path.resolve() for path in (train_output, val_output, smoke_output, report_output)
    }
    if len(resolved_outputs) != 4 or input_path.resolve() in resolved_outputs:
        raise ValueError("input, train, val, smoke and report paths must all be distinct")
    for output_path in (train_output, val_output, smoke_output, report_output):
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}; pass --overwrite to replace it"
            )

    train_capacity = train_size + max(32, math.ceil(train_size * reserve_ratio))
    val_capacity = val_size + max(32, math.ceil(val_size * reserve_ratio))
    train_heap: list[tuple[int, int, bytes, str]] = []
    val_heap: list[tuple[int, int, bytes, str]] = []
    seen: set[bytes] = set()
    invalid_reasons: Counter[str] = Counter()
    source_sha256 = hashlib.sha256()
    raw_lines = nonempty_lines = unique_records = duplicate_records = 0
    eligible_train = eligible_val = 0
    val_threshold = int(val_ratio * 2**64)

    with input_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            raw_lines += 1
            source_sha256.update(raw_line)
            if not raw_line.strip():
                continue
            nonempty_lines += 1
            if progress_every and nonempty_lines % progress_every == 0:
                print(
                    f"scanned {nonempty_lines:,} non-empty records | "
                    f"unique {unique_records:,} | duplicates {duplicate_records:,}"
                )
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except UnicodeDecodeError:
                invalid_reasons["invalid_utf8"] += 1
                continue
            except json.JSONDecodeError:
                invalid_reasons["malformed_json"] += 1
                continue

            try:
                canonical = canonical_record(record)
            except ValueError as error:
                invalid_reasons[str(error)] += 1
                continue

            canonical_bytes = canonical.encode("utf-8")
            fingerprint = hashlib.sha256(canonical_bytes).digest()
            if fingerprint in seen:
                duplicate_records += 1
                continue
            seen.add(fingerprint)
            unique_records += 1

            split_score = stable_u64(seed, "split", fingerprint)
            if split_score < val_threshold:
                eligible_val += 1
                priority = stable_u64(seed, "val-sample", fingerprint)
                _offer_candidate(
                    val_heap,
                    val_capacity,
                    priority,
                    fingerprint,
                    canonical,
                )
            else:
                eligible_train += 1
                priority = stable_u64(seed, "train-sample", fingerprint)
                _offer_candidate(
                    train_heap,
                    train_capacity,
                    priority,
                    fingerprint,
                    canonical,
                )

    if eligible_train < train_size or eligible_val < val_size:
        raise ValueError(
            "Not enough unique records in one split: "
            f"train {eligible_train:,}/{train_size:,}, val {eligible_val:,}/{val_size:,}"
        )

    train_records, train_validation = _validate_candidates(
        _ordered_candidates(train_heap),
        train_size,
        tokenizer,
        max_length,
        seed,
        empty_think_ratio,
    )
    val_records, val_validation = _validate_candidates(
        _ordered_candidates(val_heap),
        val_size,
        tokenizer,
        max_length,
        seed,
        0.0,
    )
    smoke_records = train_records[:smoke_size]

    train_sha256 = _write_jsonl_atomic(train_output, train_records, overwrite)
    val_sha256 = _write_jsonl_atomic(val_output, val_records, overwrite)
    smoke_sha256 = _write_jsonl_atomic(smoke_output, smoke_records, overwrite)

    tokenizer_sha256 = None
    if tokenizer_path is not None:
        tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()

    report: dict[str, Any] = {
        "input": str(input_path),
        "source_bytes": input_path.stat().st_size,
        "source_sha256": source_sha256.hexdigest(),
        "tokenizer": str(tokenizer_path) if tokenizer_path is not None else None,
        "tokenizer_sha256": tokenizer_sha256,
        "template_version": TEMPLATE_VERSION,
        "method": "canonical SHA-256 deduplication before stable hash split and sampling",
        "seed": seed,
        "val_ratio": val_ratio,
        "max_length": max_length,
        "train_empty_think_ratio": empty_think_ratio,
        "val_empty_think_ratio": 0.0,
        "raw_lines": raw_lines,
        "nonempty_lines": nonempty_lines,
        "invalid_records": sum(invalid_reasons.values()),
        "invalid_reasons": dict(invalid_reasons),
        "unique_records": unique_records,
        "exact_duplicates_removed": duplicate_records,
        "eligible_train_records": eligible_train,
        "eligible_val_records": eligible_val,
        "train": {
            "path": str(train_output),
            "records": len(train_records),
            "sha256": train_sha256,
            **train_validation,
        },
        "validation": {
            "path": str(val_output),
            "records": len(val_records),
            "sha256": val_sha256,
            **val_validation,
        },
        "smoke": {
            "path": str(smoke_output),
            "records": len(smoke_records),
            "sha256": smoke_sha256,
            "is_subset_of_train": True,
        },
    }
    _write_report_atomic(report_output, report, overwrite)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic train/validation/smoke MiniMind SFT subsets"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/minimind_sft/sft_t2t_mini.jsonl"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizers/minimind_bpe_16k_110k.json"),
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/processed/minimind_sft/sft_train_20k_seed42.jsonl"),
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path("data/processed/minimind_sft/sft_val_1k_seed42.jsonl"),
    )
    parser.add_argument(
        "--smoke-output",
        type=Path,
        default=Path("data/processed/minimind_sft/sft_smoke_256_seed42.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/sft_prepare_20k_1k_seed42.json"),
    )
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--val-size", type=int, default=1_000)
    parser.add_argument("--smoke-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--reserve-ratio", type=float, default=0.05)
    parser.add_argument("--empty-think-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    report = prepare_sft_subsets(
        input_path=args.input,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        train_output=args.train_output,
        val_output=args.val_output,
        smoke_output=args.smoke_output,
        report_output=args.report,
        train_size=args.train_size,
        val_size=args.val_size,
        smoke_size=args.smoke_size,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        reserve_ratio=args.reserve_ratio,
        empty_think_ratio=args.empty_think_ratio,
        seed=args.seed,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
    )

    print(f"source records: {report['nonempty_lines']:,}")
    print(
        f"unique: {report['unique_records']:,} | "
        f"duplicates removed: {report['exact_duplicates_removed']:,} | "
        f"invalid: {report['invalid_records']:,}"
    )
    print(
        f"train: {report['train']['records']:,} -> {report['train']['path']} | "
        f"val: {report['validation']['records']:,} -> "
        f"{report['validation']['path']}"
    )
    print(f"smoke: {report['smoke']['records']:,} -> {report['smoke']['path']}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
