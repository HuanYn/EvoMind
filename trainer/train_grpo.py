"""Single-GPU MiniMind-style GRPO with local rollouts.

``exact_math`` is retained solely as a deterministic algorithm test.  The
project's main route is ``minimind_rlaif``: official RLAIF prompt histories,
MiniMind-style rule rewards, and a frozen preference reward model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import torch
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_training_curves import plot_training_curves  # noqa: E402
from model.chat_template import TEMPLATE_VERSION, encode_generation_prompt  # noqa: E402
from model.config import ModelConfig  # noqa: E402
from model.grpo import completion_logps, group_advantages, grpo_policy_loss  # noqa: E402
from model.grpo_reward import score_final_integer  # noqa: E402
from model.model_minimind import MiniMindModel  # noqa: E402
from model.rlaif_reward import score_minimind_rlaif  # noqa: E402
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


def read_math_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list):
            raise ValueError(f"{path}:{line_number} must contain a messages list")
        answer = raw.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, int):
            raise ValueError(f"{path}:{line_number} must contain integer answer")
        records.append({"id": raw.get("id", str(line_number)), "messages": raw["messages"], "answer": answer})
    if not records:
        raise ValueError(f"GRPO data is empty: {path}")
    return records


def read_rlaif_records(path: Path) -> list[dict[str, object]]:
    """Read prompt-only RLAIF records after their blank assistant turn was removed."""

    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        messages = raw.get("messages") if isinstance(raw, dict) else None
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}:{line_number} must contain a non-empty messages list")
        if not isinstance(messages[-1], dict) or messages[-1].get("role") != "user":
            raise ValueError(f"{path}:{line_number} must end with the user prompt being rolled out")
        records.append({"id": raw.get("id", str(line_number)), "messages": messages})
    if not records:
        raise ValueError(f"RLAIF data is empty: {path}")
    return records


class MiniMindRewardModel:
    """Official MiniMind-compatible adapter around InternLM's reward API."""

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError("minimind_rlaif requires transformers; install it in the MLLM environment") from error
        # InternLM2 Reward ships a SentencePiece tokenizer.model.  Recent
        # Transformers may otherwise select its fast wrapper and attempt an
        # unnecessary tiktoken conversion; the official slow tokenizer is the
        # reliable cross-platform path for per-response RM scoring.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True).to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, messages: list[dict[str, object]], response: str) -> float:
        history = "\n".join(f"{message['role']}: {message['content']}" for message in messages[:-1])
        query = str(messages[-1]["content"])
        evaluation_messages = [
            {"role": "user", "content": f"{history}\nuser: {query}" if history else query},
            {"role": "assistant", "content": response},
        ]
        value = self.model.get_score(self.tokenizer, evaluation_messages)
        return float(value.item() if isinstance(value, torch.Tensor) else value)


def response_rewards(
    responses: list[str], record: dict[str, object], mode: str, reward_model: MiniMindRewardModel | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Score a rollout group and return both total rewards and interpretable parts."""

    if mode == "exact_math":
        values = [score_final_integer(text, record["answer"]).reward for text in responses]
        return torch.tensor(values, dtype=torch.float32), {"success_rate": sum(values) / len(values)}
    if reward_model is None:
        raise RuntimeError("minimind_rlaif requires a frozen reward model")
    breakdowns = [score_minimind_rlaif(text, reward_model.score(record["messages"], text)) for text in responses]
    values = [item.reward for item in breakdowns]
    return torch.tensor(values, dtype=torch.float32), {
        "positive_reward_rate": sum(item.reward > 0 for item in breakdowns) / len(breakdowns),
        "length_reward": sum(item.length_reward for item in breakdowns) / len(breakdowns),
        "think_reward": sum(item.think_reward for item in breakdowns) / len(breakdowns),
        "repetition_penalty": sum(item.repetition_penalty for item in breakdowns) / len(breakdowns),
        "reward_model_score": sum(item.reward_model_score for item in breakdowns) / len(breakdowns),
    }


def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits / temperature
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
        remove[..., 0] = False
        logits = logits.scatter(-1, sorted_indices, sorted_logits.masked_fill(remove, float("-inf")))
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


@torch.no_grad()
def rollout_group(
    model: MiniMindModel,
    tokenizer: BPETokenizer,
    messages: list[dict[str, object]],
    *,
    group_size: int,
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, list[str]]:
    """Generate G continuations and retain a mask ending exactly at EOS.

    The generation policy is the current policy before its optimizer update;
    the trainer subsequently saves its completion token log-probabilities as
    ``old_logps``, exactly as MiniMind's rollout engine contract requires.
    """
    # ``max_length`` is the model's total context window, not a prompt-only
    # limit. Reserve the requested completion budget before left-truncating
    # history; otherwise an exactly-full prompt leaves no position to roll out.
    prompt_budget = max_length - max_new_tokens
    if prompt_budget < 1:
        raise ValueError("max_new_tokens must be smaller than max_length")
    prompt = encode_generation_prompt(messages, tokenizer, max_length=prompt_budget)
    allowed_new = max_new_tokens
    ids = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0).repeat(group_size, 1)
    current = ids
    cache = None
    alive = torch.ones(group_size, dtype=torch.bool, device=device)
    completion_ids: list[torch.Tensor] = []
    completion_mask: list[torch.Tensor] = []
    model.eval()
    for _ in range(allowed_new):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits, cache = model(current, past_key_values=cache, use_cache=True)
        next_ids = sample(logits[:, -1, :].float(), temperature, top_k, top_p).squeeze(1)
        valid = alive.clone()
        next_ids = torch.where(valid, next_ids, torch.full_like(next_ids, tokenizer.pad_id))
        completion_ids.append(next_ids)
        completion_mask.append(valid)
        alive &= next_ids.ne(tokenizer.eos_id)
        current = next_ids.unsqueeze(1)
        if not alive.any():
            break
    if not completion_ids:
        raise RuntimeError("GRPO rollout produced no completion tokens")
    completion = torch.stack(completion_ids, dim=1)
    mask = torch.stack(completion_mask, dim=1)
    full_ids = torch.cat([ids, completion], dim=1)
    response_texts = [
        tokenizer.decode(row[valid].tolist())
        for row, valid in zip(completion, mask)
    ]
    return full_ids, mask, len(prompt), response_texts


def forward_completion_logps(
    model: MiniMindModel,
    full_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    prompt_length: int,
    *,
    include_router: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    full_mask = torch.cat(
        [torch.ones((full_ids.size(0), prompt_length), dtype=torch.bool, device=full_ids.device), completion_mask], dim=1
    )
    if include_router:
        logits, router_aux, _ = model(full_ids, token_mask=full_mask, return_router_loss=True)
    else:
        logits = model(full_ids, token_mask=full_mask)
        router_aux = logits.new_zeros(())
    return completion_logps(logits, full_ids, prompt_length, completion_mask.size(1)), router_aux


def one_prompt_objective(
    policy: MiniMindModel,
    reference: MiniMindModel,
    tokenizer: BPETokenizer,
    record: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    reward_model: MiniMindRewardModel | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    full_ids, completion_mask, prompt_length, responses = rollout_group(
        policy, tokenizer, record["messages"], group_size=args.num_generations,
        max_length=args.max_length, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
    )
    rewards, reward_parts = response_rewards(responses, record, args.reward_mode, reward_model)
    rewards = rewards.to(device)
    advantages = group_advantages(rewards, args.num_generations)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        old_logps, _ = forward_completion_logps(policy, full_ids, completion_mask, prompt_length, include_router=False)
        reference_logps, _ = forward_completion_logps(reference, full_ids, completion_mask, prompt_length, include_router=False)
    policy.train()
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        policy_logps, router_aux = forward_completion_logps(
            policy, full_ids, completion_mask, prompt_length, include_router=policy.config.use_moe
        )
        policy_loss, diagnostics = grpo_policy_loss(
            policy_logps, old_logps, reference_logps, advantages, completion_mask,
            epsilon=args.epsilon, beta=args.beta,
        )
        total_loss = policy_loss + policy.config.router_aux_loss_coef * router_aux
    metrics = {
        "reward": rewards.mean().item(),
        "raw_reward_std": rewards.std(unbiased=False).item(),
        "success_rate": reward_parts.get("success_rate", 0.0),
        "positive_reward_rate": reward_parts.get("positive_reward_rate", 0.0),
        "length_reward": reward_parts.get("length_reward", 0.0),
        "think_reward": reward_parts.get("think_reward", 0.0),
        "repetition_penalty": reward_parts.get("repetition_penalty", 0.0),
        "reward_model_score": reward_parts.get("reward_model_score", 0.0),
        "adv_std": advantages.std(unbiased=False).item(),
        "policy_loss": diagnostics["policy_loss"].detach().float().item(),
        "kl": diagnostics["kl"].detach().float().item(),
        "ratio": diagnostics["ratio"].detach().float().item(),
        "clip_fraction": diagnostics["clip_fraction"].detach().float().item(),
        "response_length": completion_mask.sum(dim=1).float().mean().item(),
        "router_aux": router_aux.detach().float().item(),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate(policy: MiniMindModel, tokenizer: BPETokenizer, records: list[dict[str, object]], args: argparse.Namespace, device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype, reward_model: MiniMindRewardModel | None) -> dict[str, float]:
    # Seeded stochastic validation makes checkpoint comparisons reproducible.
    policy.eval()
    totals = {"reward": 0.0, "raw_reward_std": 0.0, "success_rate": 0.0, "positive_reward_rate": 0.0, "adv_std": 0.0, "response_length": 0.0, "length_reward": 0.0, "think_reward": 0.0, "repetition_penalty": 0.0, "reward_model_score": 0.0}
    selected = records[: min(len(records), args.eval_prompts)]
    for record in selected:
        _, completion_mask, _, responses = rollout_group(
            policy, tokenizer, record["messages"], group_size=args.num_generations,
            max_length=args.max_length, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        )
        rewards, reward_parts = response_rewards(responses, record, args.reward_mode, reward_model)
        rewards = rewards.to(device)
        advantages = group_advantages(rewards, args.num_generations)
        totals["reward"] += rewards.mean().item()
        totals["raw_reward_std"] += rewards.std(unbiased=False).item()
        for key in ("success_rate", "positive_reward_rate", "length_reward", "think_reward", "repetition_penalty", "reward_model_score"):
            totals[key] += reward_parts.get(key, 0.0)
        totals["adv_std"] += advantages.std(unbiased=False).item()
        totals["response_length"] += completion_mask.sum(dim=1).float().mean().item()
    return {key: value / len(selected) for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU MiniMind-style GRPO with exact-math or official-RLAIF rewards")
    parser.add_argument("--policy-checkpoint", type=Path, required=True, help="Policy/ref initialization; use the SFT checkpoint for the official RLAIF route")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/minimind_bpe_16k_110k.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/minimind_dense_grpo_rlaif.pt"))
    parser.add_argument("--reward-mode", choices=("exact_math", "minimind_rlaif"), default="minimind_rlaif")
    parser.add_argument("--reward-model", default="models/internlm2-1_8b-reward", help="Local path or Hugging Face ID for the frozen RM; used by minimind_rlaif")
    parser.add_argument("--reward-device", choices=("cuda", "cpu"), default="cuda", help="Device for the frozen reward model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1, help="prompts per micro-batch; each prompt rolls out G candidates")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=6, help="G candidates sampled for each prompt; MiniMind official default is 6")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip radius")
    parser.add_argument("--beta", type=float, default=0.1, help="reference-KL coefficient; MiniMind official default is 0.1")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-prompts", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--curves", type=Path)
    parser.add_argument("--overwrite-history", action="store_true")
    parser.add_argument("--no-auto-plot", action="store_true")
    args = parser.parse_args()
    positive_ints = (args.epochs, args.batch_size, args.grad_accum_steps, args.num_generations, args.max_length, args.max_new_tokens, args.eval_interval, args.eval_prompts, args.save_interval)
    if min(positive_ints) < 1 or args.num_generations < 2:
        raise ValueError("GRPO integer hyperparameters must be positive and --num-generations must be at least 2")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.max_new_tokens >= args.max_length:
        raise ValueError("--max-new-tokens must be smaller than --max-length")
    if not (0 < args.temperature and args.top_k >= 0 and 0 < args.top_p <= 1 and 0 <= args.epsilon < 1 and args.beta >= 0):
        raise ValueError("invalid sampling or GRPO objective hyperparameters")
    if not (args.lr > 0 and 0 <= args.min_lr <= args.lr and args.weight_decay >= 0 and args.grad_clip > 0):
        raise ValueError("invalid optimizer hyperparameters")
    if args.reward_mode == "minimind_rlaif" and args.reward_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--reward-device cuda requested but CUDA is unavailable")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    for path in (args.policy_checkpoint, args.train_data, args.val_data, args.tokenizer):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=True)
    if "config" not in source or "model_state_dict" not in source:
        raise ValueError("--policy-checkpoint must contain config and model_state_dict")
    config = ModelConfig(**source["config"])
    tokenizer = BPETokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != config.vocab_size or args.max_length > config.max_seq_len:
        raise ValueError("tokenizer/model vocabulary or max-length is incompatible")
    reader = read_math_records if args.reward_mode == "exact_math" else read_rlaif_records
    train_records, val_records = reader(args.train_data), reader(args.val_data)
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
    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    reward_model = None
    if args.reward_mode == "minimind_rlaif":
        reward_device = torch.device(args.reward_device)
        reward_dtype = torch.float16 if reward_device.type == "cuda" else torch.float32
        reward_model = MiniMindRewardModel(args.reward_model, reward_device, reward_dtype)
    batches_per_epoch = math.ceil(len(train_records) / args.batch_size)
    total_steps = args.max_steps or args.epochs * math.ceil(batches_per_epoch / args.grad_accum_steps)
    stem = args.checkpoint.stem
    history_path = args.metrics or Path(f"artifacts/training/{stem}.metrics.jsonl")
    curve_path = args.curves or Path(f"artifacts/training/{stem}.curves.png")
    architecture = "moe" if config.use_moe else "dense"
    run_id = f"{stem}-run-{uuid4().hex[:8]}"
    history = TrainingHistoryWriter(history_path, run_id=run_id, stage="grpo", architecture=architecture, overwrite=args.overwrite_history)
    contract = {"tokenizer_sha256": sha256_file(args.tokenizer), "template_version": TEMPLATE_VERSION, "train_data_sha256": sha256_file(args.train_data), "val_data_sha256": sha256_file(args.val_data), "policy_checkpoint": str(args.policy_checkpoint), "policy_checkpoint_sha256": sha256_file(args.policy_checkpoint), "reward_mode": args.reward_mode, "reward_model": args.reward_model if args.reward_mode == "minimind_rlaif" else None, "num_generations": args.num_generations, "epsilon": args.epsilon, "beta": args.beta}
    print(f"device: {device} | policy/reference from {args.policy_checkpoint} | architecture: {architecture}")
    print(f"train prompts: {len(train_records):,} | val prompts: {len(val_records):,} | G: {args.num_generations} | reward mode: {args.reward_mode}")
    if reward_model is not None:
        print(f"frozen reward model: {args.reward_model} | device: {args.reward_device}")
    print(f"micro batch: {args.batch_size} prompts | grad accumulation: {args.grad_accum_steps} | target steps: {total_steps}")
    print(f"metrics: {history_path} | curves: {curve_path}")
    step, started = 0, time.perf_counter()
    for epoch in range(args.epochs if args.max_steps is None else 10**9):
        # Stop the outer epoch loop after --max-steps is reached. Otherwise the
        # inner loop is skipped forever and the completion/save messages never run.
        if step >= total_steps:
            break
        order = torch.randperm(len(train_records), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        cursor = 0
        while cursor < len(batches) and step < total_steps:
            micro_batches = batches[cursor:cursor + args.grad_accum_steps]
            cursor += len(micro_batches)
            optimizer.zero_grad(set_to_none=True)
            started_step = time.perf_counter()
            totals = {key: 0.0 for key in ("reward", "raw_reward_std", "success_rate", "positive_reward_rate", "length_reward", "think_reward", "repetition_penalty", "reward_model_score", "adv_std", "policy_loss", "kl", "ratio", "clip_fraction", "response_length", "router_aux")}
            prompts_seen = 0
            for micro in micro_batches:
                losses = []
                for index in micro:
                    loss, values = one_prompt_objective(policy, reference, tokenizer, train_records[index], args, device, amp_enabled, amp_dtype, reward_model)
                    losses.append(loss)
                    for key in totals:
                        totals[key] += values[key]
                    prompts_seen += 1
                scaler.scale(torch.stack(losses).mean() / len(micro_batches)).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            step += 1
            lr = cosine_lr(step, total_steps, args.warmup_steps, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            for key in totals:
                totals[key] /= prompts_seen
            should_eval = step == 1 or step % args.eval_interval == 0 or step == total_steps
            val = None
            if should_eval:
                torch.manual_seed(args.seed + step)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + step)
                val = evaluate(policy, tokenizer, val_records, args, device, amp_enabled, amp_dtype, reward_model)
                quality = val['success_rate'] if args.reward_mode == 'exact_math' else val['positive_reward_rate']
                print(f"step {step:05d} | epoch {epoch + cursor / len(batches):.3f} | lr {lr:.2e} | reward {totals['reward']:.3f} | raw reward std {totals['raw_reward_std']:.3f} | val reward {val['reward']:.3f} | val positive {quality:.3f} | KL {totals['kl']:.4f}")
            history.append(step, event="optimizer_step", epoch=epoch + cursor / len(batches), elapsed_s=time.perf_counter()-started, lr=lr, train_grpo_loss=totals["policy_loss"], train_reward=totals["reward"], train_raw_reward_std=totals["raw_reward_std"], train_success_rate=totals["success_rate"], train_positive_reward_rate=totals["positive_reward_rate"], train_length_reward=totals["length_reward"], train_think_reward=totals["think_reward"], train_repetition_penalty=totals["repetition_penalty"], train_reward_model_score=totals["reward_model_score"], train_adv_std=totals["adv_std"], train_kl=totals["kl"], train_ratio=totals["ratio"], train_clip_fraction=totals["clip_fraction"], train_response_length=totals["response_length"], val_reward=None if val is None else val["reward"], val_raw_reward_std=None if val is None else val["raw_reward_std"], val_success_rate=None if val is None else val["success_rate"], val_positive_reward_rate=None if val is None else val["positive_reward_rate"], val_length_reward=None if val is None else val["length_reward"], val_think_reward=None if val is None else val["think_reward"], val_repetition_penalty=None if val is None else val["repetition_penalty"], val_reward_model_score=None if val is None else val["reward_model_score"], val_adv_std=None if val is None else val["adv_std"], val_response_length=None if val is None else val["response_length"], router_aux_coef=config.router_aux_loss_coef if config.use_moe else None, router_aux_raw=totals["router_aux"] if config.use_moe else None, router_aux_weighted=config.router_aux_loss_coef * totals["router_aux"] if config.use_moe else None, grad_norm=float(grad_norm.detach().float().item()), prompts_per_s=prompts_seen / max(time.perf_counter()-started_step, 1e-12))
            if step % args.save_interval == 0 or step == total_steps:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.checkpoint.with_name(f".{args.checkpoint.name}.{os.getpid()}.{uuid4().hex}.tmp")
                torch.save({"checkpoint_version": 1, "stage": "grpo", "step": step, "config": asdict(config), "model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "grpo_contract": contract, "schedule": {"max_steps": total_steps, "lr": args.lr, "min_lr": args.min_lr, "warmup_steps": args.warmup_steps}, "initialization": {"policy_and_reference_from": str(args.policy_checkpoint)}, "history_state": {"run_id": run_id, "path": str(history_path), "last_logged_step": history.last_step}}, temporary)
                os.replace(temporary, args.checkpoint)
            if should_eval and not args.no_auto_plot:
                try:
                    plot_training_curves([history_path], curve_path, title=f"{run_id} — GRPO {architecture}")
                except Exception as error:
                    print(f"warning: GRPO curve update failed: {error}")
    print(f"saved GRPO checkpoint: {args.checkpoint}")
    print(f"saved metrics: {history_path}")
    if curve_path.exists():
        print(f"saved curves: {curve_path}")


if __name__ == "__main__":
    main()
