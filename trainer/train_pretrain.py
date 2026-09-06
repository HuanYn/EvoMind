"""Pretrain the MiniMind decoder-only model from prepared BPE token streams."""

import argparse
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig  # noqa: E402
from model.data import NextTokenDataset  # noqa: E402
from model.model_minimind import MiniMindModel  # noqa: E402
from model.training_history import TrainingHistoryWriter  # noqa: E402

from scripts.plot_training_curves import plot_training_curves  # noqa: E402


def load_token_stream(path: Path) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tokens = payload["token_ids"].long()
    if tokens.ndim != 1:
        raise ValueError(f"Expected a one-dimensional token stream, got {tuple(tokens.shape)}")
    return tokens, payload


def cosine_learning_rate(step: int, max_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup followed by cosine decay, indexed from optimizer step 1."""
    if warmup_steps and step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr + (base_lr - min_lr) * cosine


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled: bool, max_batches: int) -> float:
    model.eval()
    total_loss = total_tokens = 0
    for batch_index, (input_ids, labels) in enumerate(loader):
        if batch_index >= max_batches:
            break
        input_ids, labels = input_ids.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        total_loss += loss.float().item() * labels.numel()
        total_tokens += labels.numel()
    return total_loss / total_tokens


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    config,
    step: int,
    args,
    history_state: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "schedule": {
                "max_steps": args.max_steps,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "warmup_steps": args.warmup_steps,
            },
            "history_state": history_state,
        },
        path,
    )


def load_resume_checkpoint(
    path: Path, model, optimizer, scaler, config, args, device
) -> tuple[int, dict | None]:
    """Restore the state needed to continue after a completed optimizer update."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["config"] != asdict(config):
        raise ValueError("Resume checkpoint model config does not match current arguments")

    saved_schedule = checkpoint.get("schedule")
    if saved_schedule is not None:
        expected_schedule = {
            "lr": args.lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
        }
        saved_schedule_without_length = {
            key: value for key, value in saved_schedule.items() if key != "max_steps"
        }
        if saved_schedule_without_length != expected_schedule:
            raise ValueError(
                "Resume schedule does not match the checkpoint. "
                "Keep --lr, --min-lr and --warmup-steps unchanged."
            )
        saved_max_steps = saved_schedule["max_steps"]
        if saved_max_steps != args.max_steps:
            if not (args.allow_max_steps_extension and args.max_steps > saved_max_steps):
                raise ValueError(
                    "Resume --max-steps differs from the checkpoint. "
                    "For a normal resume keep it unchanged; use "
                    "--allow-max-steps-extension only for an explicit schedule extension."
                )
            print(
                f"warning: extending cosine schedule from {saved_max_steps} to {args.max_steps} steps; "
                "the learning-rate curve changes after resume"
            )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return int(checkpoint["step"]), checkpoint.get("history_state")


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMind BPE pretraining")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/minimind_bpe.pt"))
    parser.add_argument("--resume", type=Path, help="Checkpoint to continue from")
    parser.add_argument(
        "--allow-max-steps-extension",
        action="store_true",
        help="Explicitly allow increasing total steps on resume (changes cosine LR schedule)",
    )
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128, help="Training block length")
    parser.add_argument("--max-position-embeddings", type=int, default=32_768)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=2432)
    parser.add_argument("--use-moe", action="store_true", help="Replace each dense MLP with sparse MoE")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-experts-per-tok", type=int, default=1)
    parser.add_argument("--router-aux-loss-coef", type=float, default=5e-4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument(
        "--metrics",
        type=Path,
        help="Optimizer-step JSONL metrics (default: artifacts/training/<checkpoint>.metrics.jsonl)",
    )
    parser.add_argument(
        "--curves",
        type=Path,
        help="Automatically refreshed PNG dashboard (default: artifacts/training/<checkpoint>.curves.png)",
    )
    parser.add_argument(
        "--plot-smoothing-window",
        type=int,
        default=100,
        help="Trailing optimizer-step window used only to display train curves",
    )
    parser.add_argument(
        "--overwrite-history",
        action="store_true",
        help="Replace an existing metrics file when starting a new run",
    )
    parser.add_argument(
        "--no-auto-plot",
        action="store_true",
        help="Keep JSONL metrics but do not refresh a PNG at evaluation steps",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("grad-accum-steps must be at least 1")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be non-negative")
    if args.save_interval < 1:
        raise ValueError("save-interval must be at least 1")
    if args.plot_smoothing_window < 1:
        raise ValueError("plot-smoothing-window must be at least 1")
    if not 1 <= args.num_experts_per_tok <= args.num_experts:
        raise ValueError("num-experts-per-tok must be in [1, num-experts]")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    train_tokens, train_info = load_token_stream(args.train_data)
    val_tokens, val_info = load_token_stream(args.val_data)
    if train_info["vocab_size"] != val_info["vocab_size"]:
        raise ValueError("Training and validation token streams use different vocabulary sizes")

    train_loader = DataLoader(NextTokenDataset(train_tokens, args.seq_len), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(NextTokenDataset(val_tokens, args.seq_len), batch_size=args.batch_size)
    config = ModelConfig(
        vocab_size=train_info["vocab_size"],
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.max_position_embeddings,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        router_aux_loss_coef=args.router_aux_loss_coef,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    model = MiniMindModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_iter = iter(train_loader)
    routing_load_sum = (
        torch.zeros(config.n_layers, config.num_experts, device=device)
        if config.use_moe
        else None
    )
    routing_aux_sum = torch.zeros((), device=device) if config.use_moe else None
    routing_observations = 0
    start_step = 0
    saved_history_state = None
    if args.resume is not None:
        start_step, saved_history_state = load_resume_checkpoint(
            args.resume, model, optimizer, scaler, config, args, device
        )
        if start_step >= args.max_steps:
            raise ValueError("Resume checkpoint has already reached --max-steps")

    default_stem = args.checkpoint.stem
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
    if saved_history_state is not None:
        run_id = str(saved_history_state["run_id"])
    else:
        suffix = f"resume{start_step}" if args.resume is not None else "run"
        run_id = f"{default_stem}-{suffix}-{uuid4().hex[:8]}"
    history = TrainingHistoryWriter(
        history_path,
        run_id=run_id,
        stage="pretrain",
        architecture=architecture,
        checkpoint_step=start_step if args.resume is not None else None,
        overwrite=args.overwrite_history,
    )
    elapsed_offset = (
        float(saved_history_state.get("elapsed_s", 0.0))
        if saved_history_state is not None
        else 0.0
    )
    run_started = time.perf_counter()
    auto_plot_enabled = not args.no_auto_plot

    print(f"device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")
    effective_batch = args.batch_size * args.grad_accum_steps
    print(f"train tokens: {len(train_tokens):,} | val tokens: {len(val_tokens):,}")
    print(
        f"micro batch: {args.batch_size} | grad accumulation: {args.grad_accum_steps} | "
        f"effective batch: {effective_batch} | tokens / optimizer step: {effective_batch * args.seq_len:,}"
    )
    if args.resume is not None:
        print(f"resumed from: {args.resume} | completed optimizer steps: {start_step}")
    print(f"metrics: {history_path} | curves: {curve_path}")

    for step in range(start_step + 1, args.max_steps + 1):
        step_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total_loss = 0.0
        train_ce_numerator = 0.0
        train_target_tokens = 0
        step_router_aux_sum = 0.0
        step_router_load_sum = (
            torch.zeros_like(routing_load_sum) if args.use_moe else None
        )
        for _ in range(args.grad_accum_steps):
            try:
                input_ids, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, labels = next(train_iter)

            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                if args.use_moe:
                    logits, router_aux_loss, expert_fractions = model(
                        input_ids, return_router_loss=True
                    )
                else:
                    logits = model(input_ids)
                    router_aux_loss = logits.new_zeros(())
                token_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
                objective = token_loss + args.router_aux_loss_coef * router_aux_loss
                loss_for_backward = objective / args.grad_accum_steps
            scaler.scale(loss_for_backward).backward()
            target_tokens = labels.numel()
            train_ce_numerator += token_loss.detach().float().item() * target_tokens
            train_target_tokens += target_tokens
            train_total_loss += (
                objective.detach().float().item() / args.grad_accum_steps
            )
            if args.use_moe:
                routing_aux_sum += router_aux_loss.detach().float()
                expert_loads = torch.stack(expert_fractions).detach().float()
                routing_load_sum += expert_loads
                routing_observations += 1
                step_router_aux_sum += router_aux_loss.detach().float().item()
                step_router_load_sum += expert_loads

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        current_lr = cosine_learning_rate(
            step, args.max_steps, args.warmup_steps, args.lr, args.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        scaler.step(optimizer)
        scaler.update()
        step_seconds = time.perf_counter() - step_started
        train_ce = train_ce_numerator / train_target_tokens

        should_evaluate = step == 1 or step % args.eval_interval == 0 or step == args.max_steps
        should_save = step % args.save_interval == 0 or step == args.max_steps
        val_loss = None
        val_ppl = None
        if should_evaluate:
            val_loss = evaluate(model, val_loader, device, amp_enabled, args.eval_batches)
            val_ppl = math.exp(val_loss) if val_loss < math.log(float("1e308")) else None
            displayed_val_ppl = f"{val_ppl:.2f}" if val_ppl is not None else "overflow"
            print(
                f"step {step:05d} | lr {current_lr:.2e} | train loss {train_total_loss:.4f} | "
                f"val loss {val_loss:.4f} | val ppl {displayed_val_ppl}"
            )
            if args.use_moe and routing_observations:
                mean_router_aux = (routing_aux_sum / routing_observations).item()
                mean_router_load = routing_load_sum / routing_observations
                max_loads = mean_router_load.max(dim=-1).values.cpu().tolist()
                formatted_loads = ", ".join(
                    f"L{layer_index}={load:.1%}"
                    for layer_index, load in enumerate(max_loads)
                )
                print(f"router aux {mean_router_aux:.4f} | max expert load: {formatted_loads}")
                routing_aux_sum.zero_()
                routing_load_sum.zero_()
                routing_observations = 0

        step_router_aux = (
            step_router_aux_sum / args.grad_accum_steps if args.use_moe else None
        )
        step_max_loads = None
        step_router_max_load = None
        if args.use_moe:
            step_mean_load = step_router_load_sum / args.grad_accum_steps
            step_max_loads = step_mean_load.max(dim=-1).values.cpu().tolist()
            step_router_max_load = max(step_max_loads)
        tokens_seen = step * effective_batch * args.seq_len
        elapsed_s = elapsed_offset + (time.perf_counter() - run_started)
        history.append(
            step,
            event="optimizer_step",
            epoch=tokens_seen / len(train_tokens),
            elapsed_s=elapsed_s,
            samples_seen=step * effective_batch,
            tokens_seen=tokens_seen,
            supervised_tokens_seen=tokens_seen,
            lr=current_lr,
            train_ce=train_ce,
            train_total_loss=train_total_loss,
            val_ce=val_loss,
            val_ppl=val_ppl,
            grad_norm=float(grad_norm.detach().float().item()),
            tokens_per_s=(effective_batch * args.seq_len) / max(step_seconds, 1e-12),
            num_experts=config.num_experts if args.use_moe else None,
            router_aux_coef=args.router_aux_loss_coef if args.use_moe else None,
            router_aux_raw=step_router_aux,
            router_aux_weighted=(
                args.router_aux_loss_coef * step_router_aux
                if step_router_aux is not None
                else None
            ),
            router_max_load_by_layer=step_max_loads,
            router_max_load=step_router_max_load,
        )

        if should_save:
            history_state = {
                "run_id": run_id,
                "path": str(history_path),
                "last_logged_step": history.last_step,
                "elapsed_s": elapsed_s,
            }
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                scaler,
                config,
                step,
                args,
                history_state,
            )
            if not should_evaluate:
                print(f"step {step:05d} | checkpoint saved: {args.checkpoint}")
        if should_evaluate and auto_plot_enabled:
            try:
                plot_training_curves(
                    [history_path],
                    curve_path,
                    title=f"{run_id} — pretrain {architecture}",
                    smoothing_window=args.plot_smoothing_window,
                )
            except Exception as error:
                auto_plot_enabled = False
                print(
                    f"warning: automatic plotting disabled after error: {error}; "
                    f"raw metrics remain saved at {history_path}"
                )

    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved metrics: {history_path}")
    if not args.no_auto_plot and curve_path.exists():
        print(f"saved curves: {curve_path}")


if __name__ == "__main__":
    main()
