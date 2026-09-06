"""Direct Preference Optimization from a Dense MiniMind SFT checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_training_curves import plot_training_curves  # noqa: E402
from model.chat_template import TEMPLATE_VERSION  # noqa: E402
from model.config import ModelConfig  # noqa: E402
from model.dpo_data import DPODataset  # noqa: E402
from model.dpo_loss import assistant_sequence_logps, dpo_loss  # noqa: E402
from model.model_minimind import MiniMindModel  # noqa: E402
from model.tokenizer import BPETokenizer  # noqa: E402
from model.training_history import TrainingHistoryWriter  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine_lr(step: int, total: int, warmup: int, lr: float, min_lr: float) -> float:
    if warmup and step <= warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def trim_branch(input_ids: torch.Tensor, labels: torch.Tensor, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    last = input_ids.ne(pad_id).any(dim=0).nonzero(as_tuple=False)
    if last.numel() == 0:
        raise ValueError("DPO batch contains only padding")
    end = int(last[-1].item()) + 1
    if end < 2:
        raise ValueError("DPO branch is too short for shifted causal loss")
    return input_ids[:, :end], labels[:, :end]


def branch_logps(model: MiniMindModel, input_ids: torch.Tensor, labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    input_ids, labels = trim_branch(input_ids, labels, pad_id)
    model_inputs = input_ids[:, :-1]
    logits = model(model_inputs, token_mask=model_inputs.ne(pad_id))
    return assistant_sequence_logps(logits, labels)


@torch.no_grad()
def evaluate(
    policy: MiniMindModel, reference: MiniMindModel, loader: DataLoader, device: torch.device,
    *, pad_id: int, beta: float, amp_enabled: bool, amp_dtype: torch.dtype, max_batches: int,
) -> dict[str, float]:
    policy.eval()
    reference.eval()
    totals = {"loss": 0.0, "margin": 0.0, "chosen_reward": 0.0, "rejected_reward": 0.0, "accuracy": 0.0}
    examples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            policy_chosen = branch_logps(policy, batch["chosen_input_ids"], batch["chosen_labels"], pad_id)
            policy_rejected = branch_logps(policy, batch["rejected_input_ids"], batch["rejected_labels"], pad_id)
            ref_chosen = branch_logps(reference, batch["chosen_input_ids"], batch["chosen_labels"], pad_id)
            ref_rejected = branch_logps(reference, batch["rejected_input_ids"], batch["rejected_labels"], pad_id)
            loss, diagnostics = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
        count = policy_chosen.numel()
        totals["loss"] += loss.float().item() * count
        totals["margin"] += diagnostics["margin"].float().sum().item()
        totals["chosen_reward"] += diagnostics["chosen_reward"].float().sum().item()
        totals["rejected_reward"] += diagnostics["rejected_reward"].float().sum().item()
        totals["accuracy"] += diagnostics["preference_accuracy"].float().sum().item()
        examples += count
    if examples == 0:
        raise ValueError("DPO validation loader is empty")
    return {key: value / examples for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMind Dense DPO")
    parser.add_argument("--sft-checkpoint", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/minimind_bpe_16k_110k.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/minimind_dense_dpo.pt"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--curves", type=Path)
    parser.add_argument(
        "--overwrite-history",
        action="store_true",
        help="Explicitly replace an existing metrics JSONL when starting a new DPO run",
    )
    parser.add_argument("--no-auto-plot", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.grad_accum_steps, args.max_length, args.eval_interval, args.eval_batches, args.save_interval) < 1:
        raise ValueError("integer training sizes must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if not all(math.isfinite(value) for value in (args.beta, args.lr, args.min_lr, args.weight_decay, args.grad_clip)):
        raise ValueError("DPO hyperparameters must be finite")
    if args.beta <= 0 or args.lr <= 0 or args.min_lr < 0 or args.min_lr > args.lr or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("invalid DPO hyperparameters")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    for path in (args.sft_checkpoint, args.train_data, args.val_data, args.tokenizer):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = torch.load(args.sft_checkpoint, map_location="cpu", weights_only=True)
    if source.get("stage") != "sft" or "config" not in source or "model_state_dict" not in source:
        raise ValueError("--sft-checkpoint must be a complete stage='sft' checkpoint")
    config = ModelConfig(**source["config"])
    if config.use_moe:
        raise ValueError("this first DPO trainer is intentionally Dense-only")
    tokenizer = BPETokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != config.vocab_size or args.max_length > config.max_seq_len:
        raise ValueError("tokenizer/model vocabulary or max-length is incompatible")
    source_contract = source.get("training_contract", {})
    tokenizer_hash = sha256_file(args.tokenizer)
    if source_contract.get("tokenizer_sha256") not in (None, tokenizer_hash):
        raise ValueError("SFT checkpoint tokenizer SHA256 does not match")

    train_dataset = DPODataset(args.train_data, tokenizer, max_length=args.max_length)
    val_dataset = DPODataset(args.val_data, tokenizer, max_length=args.max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    policy = MiniMindModel(config).to(device)
    reference = MiniMindModel(config).to(device)
    policy.load_state_dict(source["model_state_dict"], strict=True)
    reference.load_state_dict(source["model_state_dict"], strict=True)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    del source

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    batches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    total_steps = args.max_steps or args.epochs * math.ceil(batches_per_epoch / args.grad_accum_steps)
    default_stem = args.checkpoint.stem
    history_path = args.metrics or Path(f"artifacts/training/{default_stem}.metrics.jsonl")
    curve_path = args.curves or Path(f"artifacts/training/{default_stem}.curves.png")
    run_id = f"{default_stem}-run-{uuid4().hex[:8]}"
    history = TrainingHistoryWriter(
        history_path,
        run_id=run_id,
        stage="dpo",
        architecture="dense",
        overwrite=args.overwrite_history,
    )
    contract = {
        "tokenizer_sha256": tokenizer_hash, "template_version": TEMPLATE_VERSION,
        "train_data_sha256": sha256_file(args.train_data), "val_data_sha256": sha256_file(args.val_data),
        "max_length": args.max_length, "beta": args.beta, "sft_checkpoint": str(args.sft_checkpoint),
        "sft_checkpoint_sha256": sha256_file(args.sft_checkpoint),
    }
    print(f"device: {device} | policy/reference: Dense SFT step {source_contract.get('checkpoint_step', 'unknown')}")
    print(f"train pairs: {len(train_dataset):,} | val pairs: {len(val_dataset):,} | beta: {args.beta}")
    print(f"micro batch: {args.batch_size} | grad accumulation: {args.grad_accum_steps} | target steps: {total_steps}")
    print(f"metrics: {history_path} | curves: {curve_path}")

    step = 0
    started = time.perf_counter()
    for epoch in range(args.epochs if args.max_steps is None else 10**9):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers, pin_memory=amp_enabled)
        iterator = iter(loader)
        batches_consumed = 0
        while step < total_steps:
            micro_batches = []
            for _ in range(args.grad_accum_steps):
                try:
                    micro_batches.append(next(iterator))
                    batches_consumed += 1
                except StopIteration:
                    break
            if not micro_batches:
                break
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            started_step = time.perf_counter()
            sums = {"loss": 0.0, "margin": 0.0, "chosen_reward": 0.0, "rejected_reward": 0.0, "accuracy": 0.0}
            examples = 0
            for batch in micro_batches:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    pc = branch_logps(policy, batch["chosen_input_ids"], batch["chosen_labels"], tokenizer.pad_id)
                    pr = branch_logps(policy, batch["rejected_input_ids"], batch["rejected_labels"], tokenizer.pad_id)
                    with torch.no_grad():
                        rc = branch_logps(reference, batch["chosen_input_ids"], batch["chosen_labels"], tokenizer.pad_id)
                        rr = branch_logps(reference, batch["rejected_input_ids"], batch["rejected_labels"], tokenizer.pad_id)
                    loss, diagnostics = dpo_loss(pc, pr, rc, rr, args.beta)
                    backward_loss = loss / len(micro_batches)
                scaler.scale(backward_loss).backward()
                count = pc.numel()
                sums["loss"] += loss.detach().float().item() * count
                sums["margin"] += diagnostics["margin"].detach().float().sum().item()
                sums["chosen_reward"] += diagnostics["chosen_reward"].detach().float().sum().item()
                sums["rejected_reward"] += diagnostics["rejected_reward"].detach().float().sum().item()
                sums["accuracy"] += diagnostics["preference_accuracy"].detach().float().sum().item()
                examples += count
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            step += 1
            lr = cosine_lr(step, total_steps, args.warmup_steps, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()

            should_eval = step == 1 or step % args.eval_interval == 0 or step == total_steps
            epoch_progress = epoch + batches_consumed / max(1, len(loader))
            val = None
            if should_eval:
                val = evaluate(policy, reference, DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=amp_enabled), device, pad_id=tokenizer.pad_id, beta=args.beta, amp_enabled=amp_enabled, amp_dtype=amp_dtype, max_batches=args.eval_batches)
                print(f"step {step:05d} | epoch {epoch_progress:.3f} | lr {lr:.2e} | train DPO {sums['loss']/examples:.4f} | val DPO {val['loss']:.4f} | val margin {val['margin']:.4f} | val pref acc {val['accuracy']:.3f}")
            history.append(step, event="optimizer_step", epoch=epoch_progress, elapsed_s=time.perf_counter()-started, lr=lr, train_dpo_loss=sums["loss"]/examples, train_margin=sums["margin"]/examples, train_chosen_reward=sums["chosen_reward"]/examples, train_rejected_reward=sums["rejected_reward"]/examples, train_preference_accuracy=sums["accuracy"]/examples, val_dpo_loss=None if val is None else val["loss"], val_margin=None if val is None else val["margin"], val_chosen_reward=None if val is None else val["chosen_reward"], val_rejected_reward=None if val is None else val["rejected_reward"], val_preference_accuracy=None if val is None else val["accuracy"], grad_norm=float(grad_norm.detach().float().item()), pairs_per_s=examples/max(time.perf_counter()-started_step, 1e-12))
            if step % args.save_interval == 0 or step == total_steps:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.checkpoint.with_name(f".{args.checkpoint.name}.{os.getpid()}.{uuid4().hex}.tmp")
                torch.save({"checkpoint_version": 1, "stage": "dpo", "step": step, "config": asdict(config), "model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "dpo_contract": contract, "schedule": {"max_steps": total_steps, "lr": args.lr, "min_lr": args.min_lr, "warmup_steps": args.warmup_steps}, "initialization": {"policy_and_reference_from": str(args.sft_checkpoint)}, "history_state": {"run_id": run_id, "path": str(history_path), "last_logged_step": history.last_step}}, temporary)
                os.replace(temporary, args.checkpoint)
            if should_eval and not args.no_auto_plot:
                try:
                    plot_training_curves([history_path], curve_path, title=f"{run_id} — DPO Dense")
                except Exception as error:
                    print(f"warning: DPO curve update failed: {error}")
        if step >= total_steps:
            break
    print(f"saved DPO checkpoint: {args.checkpoint}")
    print(f"saved metrics: {history_path}")
    if curve_path.exists():
        print(f"saved curves: {curve_path}")


if __name__ == "__main__":
    main()
