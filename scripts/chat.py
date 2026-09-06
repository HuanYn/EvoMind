"""Chat with a MiniMind SFT checkpoint using the shared SFT chat template."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate import sample  # noqa: E402
from model.chat_template import encode_generation_prompt  # noqa: E402
from model.config import ModelConfig  # noqa: E402
from model.model_minimind import MiniMindModel  # noqa: E402
from model.tokenizer import BPETokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one assistant reply using the MiniMind SFT template"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True, help="Current user message")
    parser.add_argument("--system", help="Optional system instruction")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--open-thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.tokenizer.is_file():
        raise FileNotFoundError(args.tokenizer)
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if args.max_length is not None and args.max_length < 1:
        raise ValueError("max-length must be positive when provided")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if args.top_k < 0:
        raise ValueError("top-k must be non-negative")
    if not math.isfinite(args.top_p) or not 0.0 < args.top_p <= 1.0:
        raise ValueError("top-p must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint is missing config or model_state_dict")
    config = ModelConfig(**checkpoint["config"])
    tokenizer = BPETokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"tokenizer vocab {tokenizer.vocab_size} != model vocab {config.vocab_size}"
        )
    max_length = args.max_length or config.max_seq_len
    if max_length > config.max_seq_len:
        raise ValueError("max-length exceeds checkpoint max_seq_len")

    conversations = []
    if args.system:
        conversations.append({"role": "system", "content": args.system})
    conversations.append({"role": "user", "content": args.prompt})
    prompt_ids = encode_generation_prompt(
        conversations,
        tokenizer,
        max_length=max_length,
        open_thinking=args.open_thinking,
    )

    model = MiniMindModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    generated_ids: list[int] = []
    past_key_values = None
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            logits, past_key_values = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = sample(
                logits[:, -1, :],
                args.temperature,
                args.top_k,
                args.top_p,
            )
            next_id = int(next_token.item())
            if next_id == tokenizer.eos_id:
                break
            generated_ids.append(next_id)
            if len(prompt_ids) + len(generated_ids) >= max_length:
                break
            input_ids = next_token

    print(
        f"device: {device} | stage: {checkpoint.get('stage', 'pretrain_legacy')} | "
        f"checkpoint step: {checkpoint.get('step', 0)}"
    )
    print(
        f"prompt tokens: {len(prompt_ids)} | generated tokens: {len(generated_ids)} | "
        f"thinking: {args.open_thinking}"
    )
    print(tokenizer.decode(generated_ids))


if __name__ == "__main__":
    main()
