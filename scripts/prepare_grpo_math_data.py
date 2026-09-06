"""Create a deterministic Chinese arithmetic prompt pool for GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


TEMPLATES = (
    "请计算 {left} {operator} {right}。请在回答最后写出整数答案。",
    "求 {left} {operator} {right} 的结果。可以简短说明，但最后必须是整数。",
    "基础算术题：{left} {operator} {right} 等于多少？最终答案写为一个整数。",
)


def _priority(seed: int, identity: str) -> bytes:
    return hashlib.sha256(f"{seed}:grpo-math:{identity}".encode("utf-8")).digest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_records(max_operand: int) -> list[dict[str, object]]:
    """Return unique addition/subtraction facts, each rendered three ways."""
    records: list[dict[str, object]] = []
    for operation, symbol in (("add", "+"), ("subtract", "-")):
        for left in range(max_operand + 1):
            for right in range(max_operand + 1):
                if operation == "subtract" and right > left:
                    continue
                answer = left + right if operation == "add" else left - right
                for template_index, template in enumerate(TEMPLATES):
                    identity = f"{operation}:{left}:{right}:template{template_index}"
                    records.append(
                        {
                            "id": identity,
                            "messages": [{"role": "user", "content": template.format(left=left, operator=symbol, right=right)}],
                            "answer": answer,
                            "operation": operation,
                            "operands": [left, right],
                            "template_index": template_index,
                        }
                    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic Chinese arithmetic prompts for GRPO")
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-operand", type=int, default=9)
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_operand < 1 or min(args.train_size, args.val_size) < 1:
        raise ValueError("max-operand, train-size and val-size must be positive")
    if len({path.resolve() for path in (args.train_output, args.val_output, args.report)}) != 3:
        raise ValueError("outputs and report must use different paths")

    candidates = build_records(args.max_operand)
    required = args.train_size + args.val_size
    if len(candidates) < required:
        raise ValueError(f"only {len(candidates)} unique prompt variants, need {required}")
    ordered = sorted(candidates, key=lambda item: _priority(args.seed, str(item["id"])))
    val_records = ordered[: args.val_size]
    train_records = ordered[args.val_size:required]
    to_jsonl = lambda records: "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    _atomic_write(args.train_output, to_jsonl(train_records))
    _atomic_write(args.val_output, to_jsonl(val_records))
    report = {
        "schema_version": 1,
        "task": "chinese_addition_subtraction_exact_final_integer",
        "reward": "1 iff the response ends with the exact integer answer; otherwise 0",
        "max_operand": args.max_operand,
        "seed": args.seed,
        "candidates": len(candidates),
        "train": {"path": str(args.train_output), "prompts": len(train_records)},
        "val": {"path": str(args.val_output), "prompts": len(val_records)},
    }
    _atomic_write(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"candidates: {len(candidates)} | train: {len(train_records)} | val: {len(val_records)}")
    print(f"reward: exact final integer only")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
