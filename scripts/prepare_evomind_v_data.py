"""Prepare a fixed original-image-grouped split from official MiniMind-V parquet."""
from pathlib import Path
import argparse
import json
import sys

import datasets  # Windows: initialize Arrow/datasets dependencies before data.py imports torch.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evomind_v.data import prepare_parquet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.0, help="Use e.g. 0.02 for a final test split never used for checkpoint selection")
    parser.add_argument("--max-rows", type=int, help="Cap scanned source rows; recorded in summary")
    parser.add_argument("--annotations", help="Optional task_type/reference_answers JSONL keyed by sample_id")
    parser.add_argument("--official-train-parquet", help="Stream original-schema accepted train rows for official V training; excludes heldout image groups")
    args = parser.parse_args()
    summary = prepare_parquet(args.parquet, args.output_dir, seed=args.seed,
                              val_fraction=args.val_fraction, max_rows=args.max_rows,
                              annotations_path=args.annotations, official_train_parquet=args.official_train_parquet,
                              test_fraction=args.test_fraction)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
