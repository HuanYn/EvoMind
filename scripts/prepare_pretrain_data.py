"""Encode one-document-per-line text into a continuous MiniMind token stream."""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BPE token IDs for MiniMind pretraining")
    parser.add_argument("--input", type=Path, required=True, help="Clean, one-document-per-line text")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    token_ids: list[int] = []
    documents = 0

    with args.input.open(encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            # Boundaries teach the model where each independent document starts and ends.
            token_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
            documents += 1

    payload = {
        "tokenizer_path": str(args.tokenizer),
        "vocab_size": tokenizer.vocab_size,
        "documents": documents,
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    print(f"documents: {documents:,}")
    print(f"tokens: {len(token_ids):,}")
    print(f"mean tokens per document: {len(token_ids) / documents:.2f}")
    print(f"special IDs: bos={tokenizer.bos_id}, eos={tokenizer.eos_id}, pad={tokenizer.pad_id}")
    print(f"saved token stream: {args.output}")


if __name__ == "__main__":
    main()
