import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.minimind.tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = CharTokenizer.build(args.text.read_text(encoding="utf-8"))
    tokenizer.save(args.output)
    print(f"saved tokenizer to: {args.output}")
    print(f"vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
