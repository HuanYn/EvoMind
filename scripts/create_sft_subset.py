"""Create a deterministic, auditable subset of an already prepared SFT train set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_priority(seed: int, canonical_json: str) -> bytes:
    return hashlib.sha256(f"{seed}:sft-subset:".encode("ascii") + canonical_json.encode("utf-8")).digest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_records(path: Path) -> list[str]:
    records: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}") from error
            canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if canonical in seen:
                raise ValueError(f"input is not deduplicated: duplicate on line {line_number}")
            seen.add(canonical)
            records.append(canonical)
    if not records:
        raise ValueError("input contains no JSON records")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic subset of an SFT train JSONL")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.size < 1:
        raise ValueError("--size must be positive")
    if args.output.resolve() == args.input.resolve():
        raise ValueError("--output must differ from --input")

    records = load_records(args.input)
    if args.size > len(records):
        raise ValueError(f"requested {args.size} records, input only has {len(records)}")

    selected = sorted(
        records,
        key=lambda canonical: stable_priority(args.seed, canonical),
    )[: args.size]
    selected_jsonl = "".join(canonical + "\n" for canonical in selected)
    atomic_write(args.output, selected_jsonl)
    report = {
        "schema_version": 1,
        "purpose": "Controlled SFT data-scale ablation: deterministic subset of an existing train split.",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "input_records": len(records),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "output_records": args.size,
        "seed": args.seed,
        "selection": "lowest SHA-256 priority over canonical JSON records",
    }
    atomic_write(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"input records: {len(records):,}")
    print(f"subset records: {args.size:,}")
    print(f"saved subset: {args.output}")
    print(f"saved report: {args.report}")


if __name__ == "__main__":
    main()
