"""Train a 16K byte-level BPE tokenizer on the cleaned Chinese corpus.

The learner runs this script. It creates a tokenizer artifact only; it does not
train MiniMind model weights.
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/wikipedia_zh_10k_clean.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tokenizers/minimind_bpe_16k.json"),
    )
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Corpus not found: {args.input}")
    if args.vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError("--vocab-size must exceed the number of special tokens")

    # BPE model stores the learned vocabulary and merge rules.
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # Repeat the corpus normalization at inference time for consistent behavior.
    tokenizer.normalizer = normalizers.NFKC()
    # Bytes provide a complete fallback alphabet for arbitrary Unicode text.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train(files=[str(args.input)], trainer=trainer)

    pad_id = tokenizer.token_to_id("<pad>")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))

    sample = "MiniMind 学习多模态大模型。"
    encoded = tokenizer.encode(sample)
    print(f"vocab size: {tokenizer.get_vocab_size():,}")
    print("special token IDs:", {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS})
    print("sample:", sample)
    print("tokens:", encoded.tokens)
    print("ids:", encoded.ids)
    print("decoded:", tokenizer.decode(encoded.ids))
    print(f"saved tokenizer: {args.output}")


if __name__ == "__main__":
    main()
