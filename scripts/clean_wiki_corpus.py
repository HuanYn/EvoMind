"""Make a clean, one-document-per-line text file for tokenizer training.

This is deliberately conservative: it normalizes Unicode, collapses whitespace,
filters short documents, and removes exact duplicates. It does not rewrite text.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean a JSONL Wikipedia sample")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/wikipedia_zh_10k.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/wikipedia_zh_10k_clean.txt"),
    )
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    kept = duplicate = too_short = 0

    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            text = normalize_text(json.loads(line)["text"])
            if len(text) < args.min_chars:
                too_short += 1
                continue

            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                duplicate += 1
                continue

            seen_hashes.add(digest)
            target.write(text + "\n")
            kept += 1

    size_mib = args.output.stat().st_size / 1024 / 1024
    print(f"kept documents: {kept:,}")
    print(f"exact duplicates removed: {duplicate:,}")
    print(f"short documents removed: {too_short:,}")
    print(f"file: {args.output}")
    print(f"size: {size_mib:.1f} MiB")


if __name__ == "__main__":
    main()
