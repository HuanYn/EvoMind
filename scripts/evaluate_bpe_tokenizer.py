"""Evaluate a trained tokenizer on held-out text, not its training corpus."""

import argparse
import json
import math
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer


def percentile(values: list[int], q: float) -> int:
    values = sorted(values)
    return values[math.ceil(q * len(values)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BPE tokenizer on held-out text")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/wikipedia_zh_val_1k_clean.txt"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizers/minimind_bpe_16k.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/tokenizer_eval_bpe_16k_val.json"),
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    unk_id = tokenizer.token_to_id("<unk>")
    doc_token_counts: list[int] = []
    total_characters = total_tokens = total_unk = round_trip_failures = 0

    with args.input.open(encoding="utf-8") as file:
        for raw_text in file:
            text = raw_text.rstrip("\n")
            if not text:
                continue

            encoding = tokenizer.encode(text)
            decoded = tokenizer.decode(encoding.ids)
            normalized = unicodedata.normalize("NFKC", text)

            total_characters += len(normalized)
            total_tokens += len(encoding.ids)
            total_unk += sum(token_id == unk_id for token_id in encoding.ids)
            round_trip_failures += decoded != normalized
            doc_token_counts.append(len(encoding.ids))

    if not doc_token_counts:
        raise ValueError("No non-empty documents found")

    report = {
        "input": str(args.input),
        "tokenizer": str(args.tokenizer),
        "vocab_size": tokenizer.get_vocab_size(),
        "documents": len(doc_token_counts),
        "characters_after_normalization": total_characters,
        "tokens": total_tokens,
        "unknown_tokens": total_unk,
        "unknown_token_rate": total_unk / total_tokens,
        "round_trip_failures_after_normalization": round_trip_failures,
        "round_trip_success_rate_after_normalization": 1 - round_trip_failures / len(doc_token_counts),
        "mean_tokens_per_document": total_tokens / len(doc_token_counts),
        "p50_tokens_per_document": percentile(doc_token_counts, 0.50),
        "p95_tokens_per_document": percentile(doc_token_counts, 0.95),
        "characters_per_token": total_characters / total_tokens,
        "tokens_per_character": total_tokens / total_characters,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"saved report: {args.report}")


if __name__ == "__main__":
    main()
