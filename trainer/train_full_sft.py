"""Supervised fine-tuning for the local MiniMind Dense or MoE checkpoint."""

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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_training_curves import plot_training_curves  # noqa: E402
from model.chat_template import (  # noqa: E402
    IGNORE_INDEX,
    TEMPLATE_VERSION,
)
from model.config import ModelConfig  # noqa: E402
from model.model_minimind import MiniMindModel  # noqa: E402
from model.sft_data import SFTDataset  # noqa: E402
from model.tokenizer import BPETokenizer  # noqa: E402
from model.training_history import TrainingHistoryWriter  # noqa: E402


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cosine_learning_rate(
    step: int,
    max_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """Optional linear warmup followed by optimizer-step cosine decay."""
    if warmup_steps and step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr + (base_lr - min_lr) * cosine


def trim_batch_right_padding(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop columns that are padding for every sample in the batch."""
    if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same [B, T] shape")
    non_padding_columns = input_ids.ne(pad_id).any(dim=0)
    positions = non_padding_columns.nonzero(as_tuple=False)
    if positions.numel() == 0:
        raise ValueError("SFT batch contains only padding")
    end = int(positions[-1].item()) + 1
    if end < 2:
        raise ValueError("SFT batch is too short for shifted causal loss")
    return input_ids[:, :end], labels[:, :end]


def shifted_sft_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Return assistant-only shifted negative log likelihood and token count.

    ``logits`` must be produced from ``input_ids[:, :-1]``. Therefore its time
    dimension aligns with ``labels[:, 1:]``.
    """
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [B, T-1, V] and labels must be [B, T]")
    shifted_labels = labels[:, 1:]
    if logits.shape[:2] != shifted_labels.shape:
        raise ValueError(
            "logits time dimension must match labels[:, 1:] for shifted SFT loss"
        )
    supervised_tokens = int(shifted_labels.ne(IGNORE_INDEX).sum().item())
    if supervised_tokens == 0:
        raise ValueError("SFT batch has no supervised assistant tokens")
    nll_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return nll_sum, supervised_tokens


def make_epoch_loader(
    dataset: SFTDataset,
    *,
    batch_size: int,
    epoch_index: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Create a reproducible shuffle order for one epoch."""
    generator = torch.Generator()
    generator.manual_seed(seed + epoch_index)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _amp_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


@torch.no_grad()
def evaluate_sft(
    model: MiniMindModel,
    loader: DataLoader,
    device: torch.device,
    *,
    pad_id: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int,
) -> tuple[float, int]:
    """Compute assistant-token-weighted validation CE."""
    model.eval()
    total_nll = 0.0
    total_supervised = 0
    for batch_index, (input_ids, labels) in enumerate(loader):
        if batch_index >= max_batches:
            break
        input_ids, labels = trim_batch_right_padding(input_ids, labels, pad_id)
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        model_inputs = input_ids[:, :-1]
        token_mask = model_inputs.ne(pad_id)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(model_inputs, token_mask=token_mask)
            nll_sum, supervised = shifted_sft_nll(logits, labels)
        total_nll += nll_sum.float().item()
        total_supervised += supervised
    if total_supervised == 0:
        raise ValueError("SFT validation produced no supervised assistant tokens")
    return total_nll / total_supervised, total_supervised


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _schedule_contract(args: argparse.Namespace, total_steps: int) -> dict[str, Any]:
    return {
        "max_steps": total_steps,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
    }


def _training_contract(
    args: argparse.Namespace,
    tokenizer: BPETokenizer,
) -> dict[str, Any]:
    return {
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "train_data_path": str(args.train_data),
        "train_data_sha256": sha256_file(args.train_data),
        "val_data_path": str(args.val_data),
        "val_data_sha256": sha256_file(args.val_data),
        "template_version": TEMPLATE_VERSION,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "seed": args.seed,
        "train_empty_think_ratio": args.empty_think_ratio,
        "train_system_prompt_ratio": args.system_prompt_ratio,
        "val_empty_think_ratio": 0.0,
        "val_system_prompt_ratio": 0.0,
        "router_aux_loss_coef": args.router_aux_loss_coef,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "amp_dtype": args.amp_dtype,
    }


def _validate_resume_contract(
    saved: dict[str, Any],
    current: dict[str, Any],
) -> None:
    immutable_fields = (
        "tokenizer_sha256",
        "vocab_size",
        "pad_id",
        "train_data_sha256",
        "val_data_sha256",
        "template_version",
        "max_length",
        "batch_size",
        "grad_accum_steps",
        "seed",
        "train_empty_think_ratio",
        "train_system_prompt_ratio",
        "val_empty_think_ratio",
        "val_system_prompt_ratio",
        "router_aux_loss_coef",
        "weight_decay",
        "grad_clip",
        "amp_dtype",
    )
    mismatches = [
        field for field in immutable_fields if saved.get(field) != current.get(field)
    ]
    if mismatches:
        raise ValueError(
            "SFT resume training contract changed: " + ", ".join(mismatches)
        )


def _validate_resume_schedule(
    saved: dict[str, Any],
    current: dict[str, Any],
    *,
    allow_extension: bool,
) -> None:
    for field in ("lr", "min_lr", "warmup_steps"):
        if saved.get(field) != current.get(field):
            raise ValueError(f"SFT resume schedule changed: {field}")
    saved_steps = int(saved["max_steps"])
    current_steps = int(current["max_steps"])
    if saved_steps == current_steps:
        return
    if not (allow_extension and current_steps > saved_steps):
        raise ValueError(
            "SFT resume max_steps changed; keep it unchanged or explicitly pass "
            "--allow-max-steps-extension when increasing it"
        )
    print(
        f"warning: extending SFT cosine schedule from {saved_steps} to "
        f"{current_steps} optimizer steps"
    )


def _save_sft_checkpoint(
    path: Path,
    *,
    model: MiniMindModel,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    config: ModelConfig,
    step: int,
    schedule: dict[str, Any],
    contract: dict[str, Any],
    trainer_state: dict[str, Any],
    initialization: dict[str, Any],
    history_state: dict[str, Any],
) -> None:
    _atomic_torch_save(
        {
            "checkpoint_version": 1,
            "stage": "sft",
            "step": step,
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "schedule": schedule,
            "training_contract": contract,
            "trainer_state": trainer_state,
            "rng_state": _rng_state(),
            "initialization": initialization,
            "history_state": history_state,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMind full supervised fine-tuning")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/processed/minimind_sft/sft_train_20k_seed42.jsonl"),
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/processed/minimind_sft/sft_val_1k_seed42.jsonl"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizers/minimind_bpe_16k_110k.json"),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        help="Start a new SFT run from model weights only",
    )
    source.add_argument(
        "--resume",
        type=Path,
        help="Continue the same SFT run with optimizer/history/data cursor",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Total optimizer steps; when omitted, epochs determines the length",
    )
    parser.add_argument("--allow-max-steps-extension", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--empty-think-ratio", type=float, default=0.2)
    parser.add_argument("--system-prompt-ratio", type=float, default=0.2)
    parser.add_argument("--router-aux-loss-coef", type=float, default=5e-4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--curves", type=Path)
    parser.add_argument("--plot-smoothing-window", type=int, default=50)
    parser.add_argument("--overwrite-history", action="store_true")
    parser.add_argument("--no-auto-plot", action="store_true")
    args = parser.parse_args()

    positive_ints = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "grad-accum-steps": args.grad_accum_steps,
        "max-length": args.max_length,
        "eval-interval": args.eval_interval,
        "eval-batches": args.eval_batches,
        "save-interval": args.save_interval,
        "plot-smoothing-window": args.plot_smoothing_window,
    }
    for name, value in positive_ints.items():
        if isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive when provided")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be non-negative")
    if not math.isfinite(args.empty_think_ratio) or not 0.0 <= args.empty_think_ratio <= 1.0:
        raise ValueError("empty-think-ratio must be in [0, 1]")
    if not math.isfinite(args.system_prompt_ratio) or not 0.0 <= args.system_prompt_ratio <= 1.0:
        raise ValueError("system-prompt-ratio must be in [0, 1]")
    if not math.isfinite(args.router_aux_loss_coef) or args.router_aux_loss_coef < 0:
        raise ValueError("router-aux-loss-coef must be non-negative")
    if (
        not math.isfinite(args.lr)
        or not math.isfinite(args.min_lr)
        or args.lr <= 0
        or args.min_lr < 0
        or args.min_lr > args.lr
    ):
        raise ValueError("learning rates must satisfy lr > 0 and 0 <= min-lr <= lr")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
        raise ValueError("grad-clip must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    for path in (args.train_data, args.val_data, args.tokenizer):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_path = args.resume or args.pretrained_checkpoint
    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    if "config" not in source_checkpoint or "model_state_dict" not in source_checkpoint:
        raise ValueError("checkpoint is missing config or model_state_dict")
    if args.resume is not None and source_checkpoint.get("stage") != "sft":
        raise ValueError("--resume requires a checkpoint with stage='sft'")
    if args.pretrained_checkpoint is not None and source_checkpoint.get("stage") == "sft":
        raise ValueError("use --resume, not --pretrained-checkpoint, for an SFT checkpoint")

    config = ModelConfig(**source_checkpoint["config"])
    if config.use_moe:
        config.router_aux_loss_coef = args.router_aux_loss_coef
    tokenizer = BPETokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"tokenizer vocab {tokenizer.vocab_size} != model vocab {config.vocab_size}"
        )
    if args.max_length > config.max_seq_len:
        raise ValueError(
            f"max-length {args.max_length} exceeds model maximum {config.max_seq_len}"
        )

    train_dataset = SFTDataset(
        args.train_data,
        tokenizer,
        max_length=args.max_length,
        seed=args.seed,
        empty_think_ratio=args.empty_think_ratio,
        system_prompt_ratio=args.system_prompt_ratio,
    )
    val_dataset = SFTDataset(
        args.val_data,
        tokenizer,
        max_length=args.max_length,
        seed=args.seed,
        empty_think_ratio=0.0,
        system_prompt_ratio=0.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = _amp_dtype(args.amp_dtype)
    pin_memory = device.type == "cuda"
    model = MiniMindModel(config).to(device)
    model.load_state_dict(source_checkpoint["model_state_dict"], strict=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    micro_batches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / args.grad_accum_steps
    )
    total_steps = args.max_steps or args.epochs * optimizer_steps_per_epoch
    schedule = _schedule_contract(args, total_steps)
    contract = _training_contract(args, tokenizer)

    start_step = 0
    trainer_state = {
        "epoch_index": 0,
        "batches_consumed_in_epoch": 0,
        "samples_seen": 0,
        "tokens_seen": 0,
        "supervised_tokens_seen": 0,
    }
    elapsed_offset = 0.0
    saved_history_state = None
    if args.resume is not None:
        _validate_resume_contract(source_checkpoint["training_contract"], contract)
        _validate_resume_schedule(
            source_checkpoint["schedule"],
            schedule,
            allow_extension=args.allow_max_steps_extension,
        )
        optimizer.load_state_dict(source_checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(source_checkpoint.get("scaler_state_dict", {}))
        start_step = int(source_checkpoint["step"])
        trainer_state = dict(source_checkpoint["trainer_state"])
        saved_history_state = source_checkpoint["history_state"]
        elapsed_offset = float(saved_history_state.get("elapsed_s", 0.0))
        _restore_rng_state(source_checkpoint.get("rng_state"))
        initialization = dict(source_checkpoint["initialization"])
        if start_step >= total_steps:
            raise ValueError("SFT resume checkpoint has already reached total steps")
    else:
        source_contract = source_checkpoint.get("training_contract") or {}
        expected_hash = source_contract.get("tokenizer_sha256")
        if expected_hash is not None and expected_hash != contract["tokenizer_sha256"]:
            raise ValueError("pretraining checkpoint tokenizer SHA256 does not match")
        if expected_hash is None:
            print(
                "warning: legacy pretraining checkpoint has no tokenizer SHA256; "
                "vocabulary size matches, and this SFT checkpoint will record the hash"
            )
        initialization = {
            "checkpoint": str(args.pretrained_checkpoint),
            "checkpoint_step": int(source_checkpoint.get("step", 0)),
            "source_stage": source_checkpoint.get("stage", "pretrain_legacy"),
        }

    checkpoint_path = args.checkpoint or (
        args.resume if args.resume is not None else Path("checkpoints/minimind_sft.pt")
    )
    default_stem = checkpoint_path.stem
    if saved_history_state is not None and args.metrics is None:
        history_path = Path(saved_history_state["path"])
    else:
        history_path = args.metrics or Path(
            f"artifacts/training/{default_stem}.metrics.jsonl"
        )
    curve_path = args.curves or Path(
        f"artifacts/training/{default_stem}.curves.png"
    )
    architecture = "moe" if config.use_moe else "dense"
    run_id = (
        str(saved_history_state["run_id"])
        if saved_history_state is not None
        else f"{default_stem}-run-{uuid4().hex[:8]}"
    )
    history = TrainingHistoryWriter(
        history_path,
        run_id=run_id,
        stage="sft",
        architecture=architecture,
        checkpoint_step=start_step if args.resume is not None else None,
        overwrite=args.overwrite_history,
    )
    auto_plot_enabled = not args.no_auto_plot
    run_started = time.perf_counter()

    del source_checkpoint
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device: {device} | amp: {args.amp_dtype if amp_enabled else 'off'} | "
        f"architecture: {architecture} | parameters: {parameter_count:,}"
    )
    print(
        f"train conversations: {len(train_dataset):,} | "
        f"val conversations: {len(val_dataset):,} | max length: {args.max_length}"
    )
    print(
        f"micro batch: {args.batch_size} | grad accumulation: "
        f"{args.grad_accum_steps} | effective batch: "
        f"{args.batch_size * args.grad_accum_steps}"
    )
    print(
        f"optimizer steps / epoch: {optimizer_steps_per_epoch:,} | "
        f"target optimizer steps: {total_steps:,}"
    )
    if args.resume is not None:
        print(f"resumed from: {args.resume} | completed optimizer steps: {start_step}")
    else:
        print(
            f"initialized model weights from: {args.pretrained_checkpoint} | "
            "optimizer and LR schedule reset for SFT"
        )
    print(f"metrics: {history_path} | curves: {curve_path}")

    step = start_step
    epoch_index = int(trainer_state["epoch_index"])
    batch_offset = int(trainer_state["batches_consumed_in_epoch"])
    samples_seen = int(trainer_state["samples_seen"])
    tokens_seen = int(trainer_state["tokens_seen"])
    supervised_seen = int(trainer_state["supervised_tokens_seen"])

    while step < total_steps:
        train_loader = make_epoch_loader(
            train_dataset,
            batch_size=args.batch_size,
            epoch_index=epoch_index,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if batch_offset < 0 or batch_offset > len(train_loader):
            raise ValueError("checkpoint data cursor is outside the current epoch")
        train_iterator = iter(train_loader)
        for _ in range(batch_offset):
            next(train_iterator)

        while batch_offset < len(train_loader) and step < total_steps:
            micro_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
            for _ in range(args.grad_accum_steps):
                if batch_offset >= len(train_loader):
                    break
                micro_batches.append(next(train_iterator))
                batch_offset += 1
            if not micro_batches:
                break

            trimmed_batches = [
                trim_batch_right_padding(input_ids, labels, tokenizer.pad_id)
                for input_ids, labels in micro_batches
            ]
            step_supervised = sum(
                int(labels[:, 1:].ne(IGNORE_INDEX).sum().item())
                for _, labels in trimmed_batches
            )
            if step_supervised == 0:
                raise ValueError("optimizer step has no supervised assistant tokens")

            step_started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            step_nll = 0.0
            step_aux_sum = 0.0
            step_samples = 0
            step_tokens = 0
            router_token_total = 0
            router_load_counts = (
                torch.zeros(
                    config.n_layers,
                    config.num_experts,
                    device=device,
                    dtype=torch.float32,
                )
                if config.use_moe
                else None
            )

            for input_ids, labels in trimmed_batches:
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                model_inputs = input_ids[:, :-1]
                token_mask = model_inputs.ne(tokenizer.pad_id)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    if config.use_moe:
                        logits, router_aux, expert_fractions = model(
                            model_inputs,
                            token_mask=token_mask,
                            return_router_loss=True,
                        )
                    else:
                        logits = model(model_inputs, token_mask=token_mask)
                        router_aux = logits.new_zeros(())
                        expert_fractions = []
                    nll_sum, supervised = shifted_sft_nll(logits, labels)
                    backward_loss = nll_sum / step_supervised
                    if config.use_moe:
                        backward_loss = backward_loss + (
                            config.router_aux_loss_coef
                            * router_aux
                            / len(trimmed_batches)
                        )
                scaler.scale(backward_loss).backward()

                step_nll += nll_sum.detach().float().item()
                step_aux_sum += router_aux.detach().float().item()
                step_samples += input_ids.size(0)
                step_tokens += int(input_ids.ne(tokenizer.pad_id).sum().item())
                if config.use_moe:
                    routed_tokens = int(token_mask.sum().item())
                    fractions = torch.stack(expert_fractions).detach().float()
                    router_load_counts += fractions * routed_tokens
                    router_token_total += routed_tokens

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            step += 1
            current_lr = cosine_learning_rate(
                step,
                total_steps,
                args.warmup_steps,
                args.lr,
                args.min_lr,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_lr
            scaler.step(optimizer)
            scaler.update()
            train_seconds = time.perf_counter() - step_started

            train_ce = step_nll / step_supervised
            mean_aux = (
                step_aux_sum / len(trimmed_batches) if config.use_moe else None
            )
            train_total_loss = train_ce + (
                config.router_aux_loss_coef * mean_aux
                if mean_aux is not None
                else 0.0
            )
            samples_seen += step_samples
            tokens_seen += step_tokens
            supervised_seen += step_supervised

            if batch_offset == len(train_loader):
                next_epoch_index = epoch_index + 1
                next_batch_offset = 0
            else:
                next_epoch_index = epoch_index
                next_batch_offset = batch_offset
            data_epoch = next_epoch_index + (
                next_batch_offset / max(1, len(train_loader))
            )

            should_evaluate = (
                step == 1
                or step % args.eval_interval == 0
                or step == total_steps
            )
            should_save = step % args.save_interval == 0 or step == total_steps
            val_ce = None
            val_ppl = None
            val_supervised = None
            if should_evaluate:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=pin_memory,
                )
                val_ce, val_supervised = evaluate_sft(
                    model,
                    val_loader,
                    device,
                    pad_id=tokenizer.pad_id,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    max_batches=args.eval_batches,
                )
                val_ppl = math.exp(val_ce) if val_ce < math.log(float("1e308")) else None
                displayed_ppl = f"{val_ppl:.2f}" if val_ppl is not None else "overflow"
                print(
                    f"step {step:05d} | epoch {data_epoch:.3f} | "
                    f"lr {current_lr:.2e} | train CE {train_ce:.4f} | "
                    f"val CE {val_ce:.4f} | val PPL {displayed_ppl}"
                )

            max_load_by_layer = None
            max_load = None
            if config.use_moe:
                mean_load = router_load_counts / router_token_total
                max_load_by_layer = mean_load.max(dim=-1).values.cpu().tolist()
                max_load = max(max_load_by_layer)

            elapsed_s = elapsed_offset + (time.perf_counter() - run_started)
            history.append(
                step,
                event="optimizer_step",
                epoch=data_epoch,
                elapsed_s=elapsed_s,
                samples_seen=samples_seen,
                tokens_seen=tokens_seen,
                supervised_tokens_seen=supervised_seen,
                lr=current_lr,
                train_ce=train_ce,
                train_total_loss=train_total_loss,
                val_ce=val_ce,
                val_ppl=val_ppl,
                val_supervised_tokens=val_supervised,
                grad_norm=float(grad_norm.detach().float().item()),
                tokens_per_s=step_tokens / max(train_seconds, 1e-12),
                supervised_tokens_per_s=(
                    step_supervised / max(train_seconds, 1e-12)
                ),
                num_experts=config.num_experts if config.use_moe else None,
                router_aux_coef=(
                    config.router_aux_loss_coef if config.use_moe else None
                ),
                router_aux_raw=mean_aux,
                router_aux_weighted=(
                    config.router_aux_loss_coef * mean_aux
                    if mean_aux is not None
                    else None
                ),
                router_max_load_by_layer=max_load_by_layer,
                router_max_load=max_load,
            )

            current_trainer_state = {
                "epoch_index": next_epoch_index,
                "batches_consumed_in_epoch": next_batch_offset,
                "samples_seen": samples_seen,
                "tokens_seen": tokens_seen,
                "supervised_tokens_seen": supervised_seen,
            }
            history_state = {
                "run_id": run_id,
                "path": str(history_path),
                "last_logged_step": history.last_step,
                "elapsed_s": elapsed_s,
            }
            if should_save:
                _save_sft_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    step=step,
                    schedule=schedule,
                    contract=contract,
                    trainer_state=current_trainer_state,
                    initialization=initialization,
                    history_state=history_state,
                )
                if not should_evaluate:
                    print(f"step {step:05d} | checkpoint saved: {checkpoint_path}")

            if should_evaluate and auto_plot_enabled:
                try:
                    plot_training_curves(
                        [history_path],
                        curve_path,
                        title=f"{run_id} — SFT {architecture}",
                        smoothing_window=args.plot_smoothing_window,
                    )
                except Exception as error:
                    auto_plot_enabled = False
                    print(
                        f"warning: automatic plotting disabled after error: {error}; "
                        f"raw metrics remain saved at {history_path}"
                    )

            trainer_state = current_trainer_state

        epoch_index = int(trainer_state["epoch_index"])
        batch_offset = int(trainer_state["batches_consumed_in_epoch"])

    print(f"saved SFT checkpoint: {checkpoint_path}")
    print(f"saved metrics: {history_path}")
    if not args.no_auto_plot and curve_path.exists():
        print(f"saved curves: {curve_path}")


if __name__ == "__main__":
    main()
