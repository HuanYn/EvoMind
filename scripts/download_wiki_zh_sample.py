"""Stream a controlled Chinese Wikipedia sample into JSONL.

Run this script yourself; it never needs to load the full dataset into memory.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Chinese Wikipedia text sample")
    parser.add_argument("--max-documents", type=int, default=10_000)
    parser.add_argument(
        "--skip-documents",
        type=int,
        default=0,
        help="Skip this many eligible documents before saving; useful for a held-out split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/wikipedia_zh_10k.jsonl"),
    )
    args = parser.parse_args()

    if args.max_documents <= 0 or args.skip_documents < 0:
        raise ValueError("--max-documents must be positive and --skip-documents non-negative")

    # streaming=True: request examples one by one instead of downloading the full corpus.
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.zh",
        split="train",
        streaming=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with args.output.open("w", encoding="utf-8") as file:
        for example in dataset:
            text = example["text"].strip()
            # Ignore redirects and extremely short pages, which add little tokenizer value.
            if len(text) < 200:
                continue

            if skipped < args.skip_documents:
                skipped += 1
                continue

            file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            kept += 1

            if kept % 1_000 == 0:
                print(f"saved {kept:,} documents")
            if kept >= args.max_documents:
                break

    size_mib = args.output.stat().st_size / 1024 / 1024
    if args.skip_documents:
        print(f"skipped eligible documents: {skipped:,}")
    print(f"done: {kept:,} documents")
    print(f"file: {args.output}")
    print(f"size: {size_mib:.1f} MiB")


if __name__ == "__main__":
    main()
