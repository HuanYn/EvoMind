"""Inspect BPE tokenization and verify encode/decode round trips."""

import argparse
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer


DEFAULT_SAMPLES = [
    "MiniMind 学习多模态大模型。",
    "视觉编码器将图像投影到语言模型空间。",
    "GRPO、MoE 和 VLA 都是后续要复现的模块。",
    "新术语：Embodied-AI_2026，罕见符号 Ω。",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a trained BPE tokenizer")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizers/minimind_bpe_16k.json"),
    )
    parser.add_argument("--text", action="append", help="Optional extra text to inspect")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    samples = DEFAULT_SAMPLES + (args.text or [])

    print(f"vocab size: {tokenizer.get_vocab_size():,}")
    for token in ["<pad>", "<bos>", "<eos>", "<unk>"]:
        print(f"{token}: {tokenizer.token_to_id(token)}")

    for index, text in enumerate(samples, start=1):
        encoding = tokenizer.encode(text)
        decoded = tokenizer.decode(encoding.ids)
        normalized = unicodedata.normalize("NFKC", text)
        print(f"\n[{index}] text: {text}")
        print("normalized text:", normalized)
        print("tokens:", encoding.tokens)
        print("ids:", encoding.ids)
        print("token count:", len(encoding.ids))
        print("decoded:", decoded)
        print("raw round trip:", decoded == text)
        print("round trip after normalization:", decoded == normalized)


if __name__ == "__main__":
    main()
