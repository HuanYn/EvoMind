"""Merge clean one-document-per-line corpora with cross-file exact deduplication."""

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge clean corpora without exact duplicates")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    total = kept = duplicates = 0

    with args.output.open("w", encoding="utf-8") as target:
        for path in args.inputs:
            source_total = source_kept = source_duplicates = 0
            with path.open(encoding="utf-8") as source:
                for line in source:
                    text = normalize_text(line)
                    if not text:
                        continue
                    total += 1
                    source_total += 1
                    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    if digest in seen_hashes:
                        duplicates += 1
                        source_duplicates += 1
                        continue
                    seen_hashes.add(digest)
                    target.write(text + "\n")
                    kept += 1
                    source_kept += 1
            print(
                f"{path}: input {source_total:,} | kept {source_kept:,} | "
                f"cross-file duplicates {source_duplicates:,}"
            )

    size_mib = args.output.stat().st_size / 1024 / 1024
    print(f"total documents read: {total:,}")
    print(f"kept documents: {kept:,}")
    print(f"exact duplicates removed: {duplicates:,}")
    print(f"file: {args.output}")
    print(f"size: {size_mib:.1f} MiB")


if __name__ == "__main__":
    main()
