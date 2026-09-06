"""Independent assistant-only held-out evaluation for a MiniMind checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainer.train_full_sft import evaluate_sft, sha256_file  # noqa: E402
from model.chat_template import TEMPLATE_VERSION  # noqa: E402
from model.config import ModelConfig  # noqa: E402
from model.model_minimind import MiniMindModel  # noqa: E402
from model.sft_data import SFTDataset  # noqa: E402
from model.tokenizer import BPETokenizer  # noqa: E402


def _atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate assistant-only CE/PPL on an SFT validation JSONL"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for path in (args.checkpoint, args.data, args.tokenizer):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, value in {
        "max-length": args.max_length,
        "batch-size": args.batch_size,
        "max-batches": args.max_batches,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    return args


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint is missing config or model_state_dict")
    config = ModelConfig(**checkpoint["config"])
    tokenizer = BPETokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"tokenizer vocab {tokenizer.vocab_size} != model vocab {config.vocab_size}"
        )
    if args.max_length > config.max_seq_len:
        raise ValueError("max-length exceeds checkpoint max_seq_len")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniMindModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = SFTDataset(
        args.data,
        tokenizer,
        max_length=args.max_length,
        empty_think_ratio=0.0,
        system_prompt_ratio=0.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    cross_entropy, supervised_tokens = evaluate_sft(
        model,
        loader,
        device,
        pad_id=tokenizer.pad_id,
        amp_enabled=device.type == "cuda",
        amp_dtype=amp_dtype,
        max_batches=args.max_batches,
    )
    perplexity = math.exp(cross_entropy)
    output_path = args.output or Path(
        "artifacts/evaluation/"
        f"{args.checkpoint.stem}_sft_{args.data.stem}.json"
    )
    report = {
        "schema_version": 1,
        "stage": checkpoint.get("stage", "pretrain_legacy"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "checkpoint_config": asdict(config),
        "data": str(args.data),
        "data_sha256": sha256_file(args.data),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "template_version": TEMPLATE_VERSION,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "conversations_evaluated": min(len(dataset), args.batch_size * args.max_batches),
        "supervised_tokens": supervised_tokens,
        "assistant_only_cross_entropy": cross_entropy,
        "assistant_only_perplexity": perplexity,
    }
    _atomic_json_save(report, output_path)
    print(
        f"device: {device} | stage: {report['stage']} | "
        f"checkpoint step: {report['checkpoint_step']}"
    )
    print(
        f"conversations: {report['conversations_evaluated']:,} | "
        f"supervised tokens: {supervised_tokens:,}"
    )
    print(f"assistant-only cross entropy: {cross_entropy:.4f}")
    print(f"assistant-only perplexity: {perplexity:.4f}")
    print(f"saved report: {output_path}")


if __name__ == "__main__":
    main()
