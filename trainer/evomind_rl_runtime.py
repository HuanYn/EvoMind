"""Opt-in, single-process local execution for the unchanged MiniMind RL objectives.

No changes to the official rollout engine, rewards, G, lengths or epoch defaults.
Checkpoints are committed only after complete optimizer updates. PPO and opt-in
GRPO/CISPO rollout reuse persist sampled data and cursors for exact continuation.
"""
from __future__ import annotations

import datasets  # noqa: F401 -- must precede torch on this Windows installation.
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from trainer.trainer_utils import LMForRewardModel

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "evomind.rl.optimizer-boundary/1"


class LocalRewardModel(LMForRewardModel):
    """The upstream scoring method, with explicit local device/tokenizer loading."""

    def __init__(self, model_path, device="cpu", dtype=torch.float32):
        model_path = Path(model_path).resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local reward model directory missing: {model_path}")
        if importlib.util.find_spec("sentencepiece") is None:
            raise RuntimeError("The local InternLM slow tokenizer requires sentencepiece in the project environment")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, use_fast=False, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            str(model_path), torch_dtype=dtype, trust_remote_code=True, local_files_only=True)
        self.model = self.model.to(device).eval().requires_grad_(False)
        original_score = self.model.get_score

        def finite_score(*score_args, **score_kwargs):
            value = original_score(*score_args, **score_kwargs)
            # Keep the original finite-score clamp/formula, but never hide an
            # invalid +/- infinity by allowing that clamp to turn it into +/-3.
            assert_finite("raw reward model score", value)
            return value

        self.model.get_score = finite_score
        self.device = device


class CPUOnlyAdamW(torch.optim.AdamW):
    """Skip Torch 2.13's accelerator health probe for strictly CPU parameters.

    This changes no optimizer mathematics. The upstream health probe can request
    a CUDA stream even for CPU-only parameters on an accelerator-enabled build.
    """

    def _accelerator_graph_capture_health_check(self):
        if any(parameter.device.type != "cpu" for group in self.param_groups for parameter in group["params"]):
            raise RuntimeError("CPUOnlyAdamW cannot manage accelerator parameters")

    def _cuda_graph_capture_health_check(self):
        self._accelerator_graph_capture_health_check()


def assert_finite(name, *values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value.detach()).all().item():
                raise FloatingPointError(f"Non-finite {name}")
        elif not math.isfinite(float(value)):
            raise FloatingPointError(f"Non-finite {name}: {value}")


def capture_rng(device="cpu"):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.device(device).type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(torch.device(device))
    return state


def restore_rng(state, device="cpu"):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        if torch.device(device).type != "cuda":
            raise ValueError("Cannot resume a CUDA RNG contract on a CPU device")
        torch.cuda.set_rng_state(state["cuda"].cpu(), torch.device(device))


def seed_rng(seed, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    # CPU generator directly: no CUDA discovery or initialization for CPU tests.
    torch.random.default_generator.manual_seed(seed)
    if torch.device(device).type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: device_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [device_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(device_tree(v, device) for v in value)
    return value


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_runtime_arguments(parser):
    parser.add_argument("--evomind_runtime", action="store_true", help="Opt into audited local optimizer-boundary execution")
    parser.add_argument("--max_steps", type=int, default=0, help="Optimizer updates in THIS invocation; 0 completes the configured epochs")
    parser.add_argument("--init_dir", default="../out", help="Read-only initial policy/reference directory")
    parser.add_argument("--tokenizer_path", default="../model")
    parser.add_argument("--resume_dir", default=None, help="Owned state directory; defaults to save_dir/<save_weight>_runtime")
    parser.add_argument("--resume", default=None, help="Explicit trusted optimizer-boundary checkpoint")
    parser.add_argument("--reward_device", default=None, help="Independent reward device; defaults to --device")
    parser.add_argument("--reward_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio_diagnostics", action="store_true",
                        help="Record GRPO suppression and CISPO importance-cap rates for an isolated comparison")
    parser.add_argument("--updates_per_rollout", type=int, default=1,
                        help="Experimental only: optimizer updates that reuse one frozen rollout; 1 preserves the main recipe")


def masked_completion_logps(model, outputs, full_mask, positions):
    result = model(outputs, attention_mask=full_mask)
    logps = F.log_softmax(result.logits[:, :-1, :], dim=-1).gather(
        2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, positions)
    return logps, getattr(result, "aux_loss", None)


def grpo_objective(logps, old_logps, ref_logps, rewards, mask, *, generations,
                   beta, loss_type, epsilon, epsilon_high):
    grouped = rewards.view(-1, generations)
    mean = grouped.mean(dim=1).repeat_interleave(generations)
    std = grouped.std(dim=1, unbiased=False).repeat_interleave(generations)
    advantages = (rewards - mean) / (std + 1e-4)
    delta = ref_logps - logps
    kl = delta.exp() - delta - 1
    ratio = (logps - old_logps).exp()
    if loss_type == "cispo":
        token_loss = -(ratio.clamp(max=epsilon_high).detach() * advantages[:, None] * logps - beta * kl)
    else:
        token_loss = -(torch.minimum(ratio * advantages[:, None],
                                    ratio.clamp(1 - epsilon, 1 + epsilon) * advantages[:, None]) - beta * kl)
    loss = ((token_loss * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
    return loss, advantages


def grpo_cispo_ratio_statistics(logps, old_logps, ref_logps, advantages, mask, *, epsilon, epsilon_high):
    """Calculate comparable token-level GRPO/CISPO update-utilization metrics."""
    valid = mask.bool()
    if not valid.any().item():
        raise ValueError("Ratio diagnostics require at least one valid completion token")
    # Select before exponentiation/reduction: padding must not contribute,
    # including padding sentinels whose exponentials would overflow.
    selected_logps = logps.masked_select(valid)
    ratio = (selected_logps - old_logps.masked_select(valid)).exp()
    selected_advantages = advantages[:, None].expand_as(mask).masked_select(valid)
    grpo_suppressed = ((selected_advantages > 0) & (ratio > 1 + epsilon)) | (
        (selected_advantages < 0) & (ratio < 1 - epsilon))
    cispo_capped = ratio > epsilon_high
    delta = ref_logps.masked_select(valid) - selected_logps
    kl_penalty = delta.exp() - delta - 1
    selected_ratio = ratio.float()
    return {
        "grpo_suppressed_rate": float(grpo_suppressed.float().mean()),
        "cispo_capped_rate": float(cispo_capped.float().mean()),
        "ratio_mean": float(selected_ratio.mean()),
        "ratio_p95": float(torch.quantile(selected_ratio, 0.95)),
        "ratio_max": float(selected_ratio.max()),
        "reference_logp_gap": float(delta.mean()),
        "kl_penalty": float(kl_penalty.mean()),
    }


def ppo_objective(logps, old_logps, ref_logps, values, old_values, advantages, returns,
                  policy_mask, value_mask, *, clip_epsilon, kl_coef, cliprange_value):
    log_ratio = logps - old_logps
    ratio = log_ratio.exp()
    denom = policy_mask.sum().clamp(min=1)
    approx_kl = (0.5 * log_ratio.square() * policy_mask).sum() / denom
    delta = ref_logps - logps
    kl_ref = ((delta.exp() - delta - 1) * policy_mask).sum() / denom
    policy = (torch.maximum(-advantages * ratio,
                            -advantages * ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)) * policy_mask).sum() / denom + kl_coef * kl_ref
    clipped_value = values.clamp(old_values - cliprange_value, old_values + cliprange_value)
    value_loss = 0.5 * (torch.maximum((values - returns).square(), (clipped_value - returns).square()) * value_mask).sum() / value_mask.sum().clamp(min=1)
    return policy, value_loss, approx_kl, kl_ref


def completion_state(result, tokenizer, device):
    outputs = result.output_ids.to(device)
    completion_ids = result.completion_ids.to(device)
    if not completion_ids.size(1):
        raise ValueError("Empty rollout cannot define the official RL objective")
    positions = result.prompt_lens.to(device)[:, None] - 1 + torch.arange(completion_ids.size(1), device=device)[None]
    pad_mask = result.completion_mask.to(device).bool()
    eos = completion_ids.eq(tokenizer.eos_token_id) & pad_mask
    lengths = pad_mask.sum(1)
    lengths = torch.where(eos.any(1), eos.int().argmax(1) + 1, lengths).long().clamp(min=1)
    mask = ((torch.arange(completion_ids.size(1), device=device)[None] < lengths[:, None]) & pad_mask).float()
    return {"outputs": outputs, "full_mask": (outputs != tokenizer.pad_token_id).long(),
            "positions": positions, "mask": mask, "lengths": lengths,
            "valid": pad_mask.sum(1) > 0, "old_logps": result.per_token_logps.to(device).detach()}


def ppo_advantages(old_values, old_logps, rewards, mask, lengths, valid, gamma, lam):
    token_rewards = torch.zeros_like(old_logps)
    indices = torch.arange(len(rewards), device=rewards.device)[valid]
    token_rewards[indices, (lengths - 1)[valid]] += rewards[valid]
    last = torch.zeros(len(rewards), device=rewards.device)
    reversed_adv = []
    for t in reversed(range(old_values.size(1))):
        next_value = old_values[:, t + 1] if t < old_values.size(1) - 1 else 0.0
        delta = token_rewards[:, t] + gamma * next_value - old_values[:, t]
        last = delta + gamma * lam * last
        reversed_adv.append(last)
    advantages = torch.stack(reversed_adv[::-1], dim=1)
    returns = advantages + old_values
    mean = (advantages * mask).sum() / mask.sum().clamp(min=1)
    variance = ((advantages - mean).square() * mask).sum() / mask.sum().clamp(min=1)
    normalized = (advantages - mean) * torch.rsqrt(variance + 1e-8) * mask
    return normalized, returns


def checked_update(models, optimizers, schedulers, grad_clip):
    for model in models:
        for parameter in model.parameters():
            assert_finite("gradient before optimizer update", parameter.grad)
        if grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
            assert_finite("gradient norm", norm)
    for optimizer in optimizers:
        optimizer.step()
    for scheduler in schedulers:
        scheduler.step()
    for model in models:
        for parameter in model.parameters():
            assert_finite("parameter after optimizer update", parameter)
    for optimizer in optimizers:
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    assert_finite("optimizer state", value)
        for group in optimizer.param_groups:
            assert_finite("learning rate", group["lr"])
        optimizer.zero_grad(set_to_none=True)


class TrainingSession:
    """Single owned run and its complete, full-precision update-boundary state."""

    def __init__(self, args, algorithm, models, optimizers, schedulers, contract, *, total_batches):
        self.args, self.algorithm, self.models = args, algorithm, models
        self.optimizers, self.schedulers, self.contract = optimizers, schedulers, contract
        self.total_batches = total_batches
        self.directory = Path(args.resume_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "latest.resume.pt"
        self.started = time.perf_counter()
        self.elapsed_before = 0.0
        self.updates = 0
        self.invocation_updates = 0
        self.invocation_id = str(time.time_ns())
        self.resume_start_optimizer_update = 0
        self.checkpoint_optimizer_update = 0
        self.cursor = {"epoch": 0, "next_batch": 0, "order": None, "pending": None, "ppo_accum_counter": 0}
        resume = args.resume or (str(self.path) if args.from_resume else None)
        if resume:
            state = torch.load(resume, map_location="cpu", weights_only=False)
            if state.get("format") != FORMAT or state.get("algorithm") != algorithm:
                raise ValueError("Not a compatible evomind optimizer-boundary checkpoint")
            if state.get("probe_only") is not (args.max_steps > 0):
                raise ValueError("Probe/full checkpoint identities cannot be promoted or changed on resume")
            if state["contract"] != contract:
                raise ValueError("Resume input/config/code contract changed; use a separate experiment")
            for name, model in models.items():
                model.load_state_dict(state["models"][name], strict=True)
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(state["optimizers"][name])
            for name, scheduler in schedulers.items():
                scheduler.load_state_dict(state["schedulers"][name])
            self.cursor = device_tree(state["cursor"], args.device)
            # Epoch order stays on CPU, so indexing does not allocate a GPU tensor per item.
            if isinstance(self.cursor["order"], torch.Tensor):
                self.cursor["order"] = self.cursor["order"].cpu()
            self.updates = state["optimizer_updates"]
            self.resume_start_optimizer_update = self.updates
            self.checkpoint_optimizer_update = self.updates
            self.elapsed_before = state.get("elapsed_seconds", 0.0)
            restore_rng(state["rng"], args.device)
        elif self.path.exists() or (self.directory / "summary.json").exists() or (self.directory / "metrics.jsonl").exists():
            raise FileExistsError("Existing run requires explicit --resume or --from_resume 1")
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        atomic_json({"algorithm": algorithm, "contract": contract, "invocation": vars(args)}, self.directory / "run_config.json")
        self.write_summary("running", None)

    def elapsed(self):
        return self.elapsed_before + time.perf_counter() - self.started

    def contract_hash(self):
        return hashlib.sha256(json.dumps(self.contract, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

    def normalize_cursor(self):
        if self.cursor["pending"] is None and self.cursor["next_batch"] >= self.total_batches:
            self.cursor.update(epoch=self.cursor["epoch"] + 1, next_batch=0, order=None, ppo_accum_counter=0)

    def exhausted(self):
        return self.cursor["epoch"] >= self.args.epochs

    def limit_reached(self):
        return self.args.max_steps > 0 and self.invocation_updates >= self.args.max_steps

    def commit(self, metric):
        self.normalize_cursor()
        self.updates += 1
        self.invocation_updates += 1
        metric = {**metric, "optimizer_update": self.updates, "invocation_update": self.invocation_updates,
                  "invocation_id": self.invocation_id,
                  "resume_start_optimizer_update": self.resume_start_optimizer_update,
                  "elapsed_seconds": self.elapsed(), "learning_rates": {
                      name: [group["lr"] for group in optimizer.param_groups]
                      for name, optimizer in self.optimizers.items()}}
        for key, value in metric.items():
            if isinstance(value, (float, int)):
                assert_finite(key, value)
        if torch.device(self.args.device).type == "cuda":
            metric["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.args.device)
            metric["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.args.device)
        # No live graph or pending gradients may enter an update-boundary file.
        for model in self.models.values():
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Checkpoint requested before optimizer zero_grad")
        checkpoint_due = self.updates % self.args.save_interval == 0 or self.exhausted() or self.limit_reached()
        metric["checkpoint_committed"] = checkpoint_due
        if checkpoint_due:
            state = {"format": FORMAT, "algorithm": self.algorithm, "contract": self.contract,
                     "contract_sha256": self.contract_hash(), "probe_only": self.args.max_steps > 0,
                     "models": {name: cpu_tree(model.state_dict()) for name, model in self.models.items()},
                     "optimizers": {name: cpu_tree(opt.state_dict()) for name, opt in self.optimizers.items()},
                     "schedulers": {name: scheduler.state_dict() for name, scheduler in self.schedulers.items()},
                     "cursor": cpu_tree(self.cursor), "rng": capture_rng(self.args.device),
                     "optimizer_updates": self.updates, "elapsed_seconds": self.elapsed(), "last_metric": metric}
            atomic_torch_save(state, self.path)
            self.checkpoint_optimizer_update = self.updates
            self.write_summary("running", None)
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metric, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        print(json.dumps(metric, ensure_ascii=False, allow_nan=False), flush=True)

    def export_actor(self):
        suffix = "_moe" if self.args.use_moe else ""
        path = Path(self.args.save_dir) / f"{self.args.save_weight}_{self.args.hidden_size}{suffix}.pth"
        atomic_torch_save({key: value.detach().half().cpu() for key, value in self.models["actor"].state_dict().items()}, path)
        return str(path.resolve())

    def write_summary(self, status, reason, *, final_checkpoint=None, error=None):
        value = {"schema_version": "evomind.rl.run/1", "status": status, "algorithm": self.algorithm,
                 "loss_type": getattr(self.args, "loss_type", None),
                 "contract_sha256": self.contract_hash(),
                 "planned_optimizer_updates": self.contract.get("planned_optimizer_steps"),
                 "planned_optimizer_updates_note": "Nominal budget; PPO preserves upstream KL early-stop and accumulation scheduling",
                 "probe_only": self.args.max_steps > 0, "stop_reason": reason,
                 "optimizer_updates": self.updates, "invocation_optimizer_updates": self.invocation_updates,
                 "invocation_id": self.invocation_id,
                 "resume_start_optimizer_update": self.resume_start_optimizer_update,
                 "checkpoint_optimizer_update": self.checkpoint_optimizer_update,
                 "save_interval_optimizer_updates": self.args.save_interval,
                 "recovery_policy": "Resume the last saved complete update only; a crash can lose up to save_interval-1 updates. PPO and opt-in GRPO/CISPO reuse restore their saved pending rollout.",
                 "metrics_policy": "Append-only invocation records. On resume, prior tail above resume_start_optimizer_update is historical/discarded, not part of the final model trajectory.",
                 "requested_max_steps": self.args.max_steps, "configured_epochs": self.args.epochs,
                 "cursor_epoch": self.cursor["epoch"], "cursor_next_batch": self.cursor["next_batch"],
                 "pending_rollout": self.cursor["pending"] is not None,
                 "pending_ppo_rollout": self.algorithm == "ppo" and self.cursor["pending"] is not None,
                 "resume_checkpoint": str(self.path) if self.path.exists() else None,
                 "final_checkpoint": final_checkpoint, "elapsed_seconds": self.elapsed(),
                 "elapsed_scope": "training-loop wall time including checkpoints; excludes initial model/input loading",
                 "reward_device": self.args.reward_device, "reward_dtype": self.args.reward_dtype,
                 "peak_allocated_bytes": (torch.cuda.max_memory_allocated(self.args.device)
                                          if torch.device(self.args.device).type == "cuda" else None),
                 "peak_reserved_bytes": (torch.cuda.max_memory_reserved(self.args.device)
                                         if torch.device(self.args.device).type == "cuda" else None),
                 "error": error}
        atomic_json(value, self.directory / "summary.json")
        return value


def _context(args):
    return (torch.autocast("cuda", dtype=getattr(torch, args.dtype))
            if torch.device(args.device).type == "cuda" else nullcontext())


def _ensure_epoch_order(session, dataset):
    cursor = session.cursor
    if cursor["order"] is None:
        seed_rng(session.args.seed + cursor["epoch"], session.args.device)
        cursor["order"] = torch.randperm(len(dataset)).tolist()


def _next_batch(session, dataset):
    _ensure_epoch_order(session, dataset)
    position = session.cursor["next_batch"] * session.args.batch_size
    selected = session.cursor["order"][position:position + session.args.batch_size]
    rows = [dataset[int(index)] for index in selected]
    session.cursor["next_batch"] += 1
    return [row["prompt"] for row in rows]


def run_grpo_prepared(session, *, actor, reference, tokenizer, dataset, engine, reward_fn):
    args = session.args
    actor.train()
    reference.eval()
    # The production recipe deliberately consumes a new rollout per update.  This
    # opt-in branch is a mechanism ablation: it replays the *same* frozen
    # rollout after the actor changes, so GRPO clipping / CISPO capping can be
    # observed.  Pending state is checkpointed so a resume cannot resample an
    # unfinished replay group.
    if getattr(args, "updates_per_rollout", 1) > 1:
        return run_grpo_rollout_reuse(session, actor=actor, reference=reference, tokenizer=tokenizer,
                                      dataset=dataset, engine=engine, reward_fn=reward_fn)
    microsteps = 0
    while not session.exhausted():
        epoch = session.cursor["epoch"]
        prompts = _next_batch(session, dataset)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                           padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            inputs["input_ids"] = inputs["input_ids"][:, -args.max_seq_len:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -args.max_seq_len:]
        result = engine.rollout(prompt_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                num_generations=args.num_generations, max_new_tokens=args.max_gen_len, temperature=0.8)
        state = completion_state(result, tokenizer, args.device)
        expected = len(prompts) * args.num_generations
        if len(result.completions) != expected or state["outputs"].size(0) != expected:
            raise ValueError("Rollout did not preserve all generations in every reward group")
        rewards = reward_fn(prompts, result.completions).to(args.device)
        assert_finite("rollout log probabilities/rewards", state["old_logps"], rewards)
        with _context(args):
            logps, aux = masked_completion_logps(actor, state["outputs"], state["full_mask"], state["positions"])
        with torch.no_grad():
            ref_logps, _ = masked_completion_logps(reference, state["outputs"], state["full_mask"], state["positions"])
        aux = aux if args.use_moe else logps.new_zeros(())
        policy, advantages = grpo_objective(
            logps, state["old_logps"], ref_logps, rewards, state["mask"], generations=args.num_generations,
            beta=args.beta, loss_type=args.loss_type, epsilon=args.epsilon, epsilon_high=args.epsilon_high)
        loss = (policy + aux) / args.accumulation_steps
        assert_finite("GRPO/CISPO loss/log probabilities/advantages", loss, logps, ref_logps, advantages)
        loss.backward()
        microsteps += 1
        metric = {"epoch": epoch, "batch": session.cursor["next_batch"], "loss": float(policy.detach() + aux.detach()),
                  "policy_loss": float(policy.detach()), "aux_loss": float(aux.detach()),
                  "reward": float(rewards.mean()), "response_length": float(state["mask"].sum(1).mean()),
                  "advantage_population_std": float(advantages.std(unbiased=False))}
        if getattr(args, "ratio_diagnostics", False):
            ratio_stats = grpo_cispo_ratio_statistics(
                logps.detach(), state["old_logps"], ref_logps.detach(), advantages.detach(), state["mask"],
                epsilon=args.epsilon, epsilon_high=args.epsilon_high)
            metric.update(ratio_stats)
            metric["native_intervention_rate"] = (
                ratio_stats["cispo_capped_rate"] if args.loss_type == "cispo"
                else ratio_stats["grpo_suppressed_rate"])
        end_epoch = session.cursor["next_batch"] == session.total_batches
        # Preserve upstream tail scaling (/ accumulation_steps), including a short final window.
        if microsteps % args.accumulation_steps == 0 or end_epoch:
            checked_update([actor], list(session.optimizers.values()), list(session.schedulers.values()), args.grad_clip)
            session.commit(metric)
            if session.limit_reached():
                return
        if end_epoch:
            microsteps = 0
        # Torch engine holds the same actor object. No stale remote snapshot exists.
        del result, state, inputs, logps, ref_logps, loss, policy, advantages, rewards, aux


def _prepare_grpo_rollout_reuse(session, *, tokenizer, dataset, engine, reward_fn, reference):
    """Sample, score and freeze one complete group for a multi-update ablation."""
    args = session.args
    epoch = session.cursor["epoch"]
    prompts = _next_batch(session, dataset)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                       padding_side="left", add_special_tokens=False).to(args.device)
    if args.max_seq_len:
        inputs["input_ids"] = inputs["input_ids"][:, -args.max_seq_len:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -args.max_seq_len:]
    result = engine.rollout(prompt_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                            num_generations=args.num_generations, max_new_tokens=args.max_gen_len, temperature=0.8)
    state = completion_state(result, tokenizer, args.device)
    expected = len(prompts) * args.num_generations
    if len(result.completions) != expected or state["outputs"].size(0) != expected:
        raise ValueError("Rollout did not preserve all generations in every reward group")
    rewards = reward_fn(prompts, result.completions).to(args.device)
    assert_finite("rollout-reuse log probabilities/rewards", state["old_logps"], rewards)
    with torch.no_grad():
        ref_logps, _ = masked_completion_logps(reference, state["outputs"], state["full_mask"], state["positions"])
    grouped = rewards.view(-1, args.num_generations)
    means = grouped.mean(dim=1).repeat_interleave(args.num_generations)
    stds = grouped.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
    advantages = (rewards - means) / (stds + 1e-4)
    assert_finite("rollout-reuse frozen state", ref_logps, advantages)
    state.update(epoch=epoch, prompts=prompts, rewards=rewards.detach(), ref_logps=ref_logps.detach(),
                 advantages=advantages.detach(), replay_index=0,
                 replay_total=args.updates_per_rollout)
    # Cursor.pending blocks epoch advancement in TrainingSession.commit and is
    # serialized at each optimizer boundary for exact replay-group recovery.
    session.cursor["pending"] = state
    del result, inputs, rewards, ref_logps, advantages


def run_grpo_rollout_reuse(session, *, actor, reference, tokenizer, dataset, engine, reward_fn):
    """Mechanism-only GRPO/CISPO comparison with K optimizer updates per rollout."""
    args = session.args
    while not session.exhausted():
        if session.cursor["pending"] is None:
            _prepare_grpo_rollout_reuse(session, tokenizer=tokenizer, dataset=dataset, engine=engine,
                                        reward_fn=reward_fn, reference=reference)
        state = session.cursor["pending"]
        if state.get("replay_total") != args.updates_per_rollout:
            raise ValueError("Resume rollout-reuse count differs from this invocation")
        replay_index = int(state["replay_index"]) + 1
        if replay_index < 1 or replay_index > args.updates_per_rollout:
            raise RuntimeError("Invalid stored rollout-reuse cursor")
        with _context(args):
            logps, aux = masked_completion_logps(actor, state["outputs"], state["full_mask"], state["positions"])
        aux = aux if args.use_moe else logps.new_zeros(())
        policy, advantages = grpo_objective(
            logps, state["old_logps"], state["ref_logps"], state["rewards"], state["mask"],
            generations=args.num_generations, beta=args.beta, loss_type=args.loss_type,
            epsilon=args.epsilon, epsilon_high=args.epsilon_high)
        loss = policy + aux
        assert_finite("rollout-reuse GRPO/CISPO loss", loss, logps, advantages)
        loss.backward()
        ratio_stats = grpo_cispo_ratio_statistics(
            logps.detach(), state["old_logps"], state["ref_logps"], advantages.detach(), state["mask"],
            epsilon=args.epsilon, epsilon_high=args.epsilon_high)
        metric = {
            "epoch": state["epoch"], "batch": session.cursor["next_batch"],
            "loss": float(policy.detach() + aux.detach()), "policy_loss": float(policy.detach()),
            "aux_loss": float(aux.detach()), "reward": float(state["rewards"].mean()),
            "response_length": float(state["mask"].sum(1).mean()),
            "advantage_population_std": float(advantages.std(unbiased=False)),
            "rollout_reuse_total": args.updates_per_rollout,
            "rollout_reuse_index": replay_index,
            **ratio_stats,
            "native_intervention_rate": (ratio_stats["cispo_capped_rate"] if args.loss_type == "cispo"
                                         else ratio_stats["grpo_suppressed_rate"]),
        }
        final_replay = replay_index == args.updates_per_rollout
        if final_replay:
            # This exact replay has been consumed.  The checkpoint may now
            # advance the dataset cursor and does not retain its rollout.
            session.cursor["pending"] = None
        else:
            state["replay_index"] = replay_index
        checked_update([actor], list(session.optimizers.values()), list(session.schedulers.values()), args.grad_clip)
        session.commit(metric)
        if session.limit_reached():
            return
        del logps, aux, policy, advantages, loss


def _prepare_ppo_rollout(session, *, actor, critic, reference, tokenizer, dataset, engine, reward_fn):
    args = session.args
    prompts = _next_batch(session, dataset)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=args.max_seq_len, padding_side="left").to(args.device)
    result = engine.rollout(prompt_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                            num_generations=1, max_new_tokens=args.max_gen_len, temperature=0.8)
    state = completion_state(result, tokenizer, args.device)
    if len(result.completions) != len(prompts):
        raise ValueError("PPO rollout count differs from its prompt batch")
    rewards = reward_fn(prompts, result.completions).to(args.device)
    with torch.no_grad():
        old_values = critic(input_ids=state["outputs"], attention_mask=state["full_mask"]).gather(1, state["positions"]) * state["mask"]
        ref_logps, _ = masked_completion_logps(reference, state["outputs"], state["full_mask"], state["positions"])
        advantages, returns = ppo_advantages(old_values, state["old_logps"], rewards, state["mask"],
                                             state["lengths"], state["valid"], args.gamma, args.lam)
    assert_finite("PPO rollout state", old_values, state["old_logps"], ref_logps, advantages, returns, rewards)
    state.update(old_values=old_values, ref_logps=ref_logps, advantages=advantages, returns=returns,
                 rewards=rewards, ppo_epoch=0, permutation=None, mini_start=0, stop_ppo=False)
    return state


def run_ppo_prepared(session, *, actor, critic, reference, tokenizer, dataset, engine, reward_fn):
    args = session.args
    actor.train()
    critic.train()
    reference.eval()
    while not session.exhausted():
        if session.cursor["pending"] is None:
            session.cursor["pending"] = _prepare_ppo_rollout(
                session, actor=actor, critic=critic, reference=reference, tokenizer=tokenizer,
                dataset=dataset, engine=engine, reward_fn=reward_fn)
        pending = session.cursor["pending"]
        batch_size = pending["outputs"].size(0)
        mini_size = max(1, min(args.mini_batch_size, batch_size))
        if pending["permutation"] is None:
            pending["permutation"] = torch.randperm(batch_size, device=args.device)
        begin = pending["mini_start"]
        indices = pending["permutation"][begin:begin + mini_size]
        values = critic(input_ids=pending["outputs"][indices], attention_mask=pending["full_mask"][indices]).gather(1, pending["positions"][indices])
        with _context(args):
            logps, aux = masked_completion_logps(actor, pending["outputs"][indices], pending["full_mask"][indices], pending["positions"][indices])
        aux = aux if args.use_moe else logps.new_zeros(())
        policy, value_loss, approx_kl, kl_ref = ppo_objective(
            logps, pending["old_logps"][indices], pending["ref_logps"][indices], values,
            pending["old_values"][indices], pending["advantages"][indices], pending["returns"][indices],
            pending["mask"][indices], pending["mask"][indices], clip_epsilon=args.clip_epsilon,
            kl_coef=args.kl_coef, cliprange_value=args.cliprange_value)
        assert_finite("PPO losses/log probabilities/values", policy, value_loss, approx_kl, kl_ref, logps, values)
        if approx_kl.detach() > args.early_stop_kl:
            pending["stop_ppo"] = True
        combined = policy + args.vf_coef * value_loss + aux
        # Preserve upstream: zero loss on KL stop, finish current PPO epoch's minibatches.
        loss = combined * 0 if pending["stop_ppo"] else combined / args.accumulation_steps
        assert_finite("PPO combined loss", loss)
        loss.backward()
        session.cursor["ppo_accum_counter"] += 1
        pending["mini_start"] += mini_size
        if pending["mini_start"] >= batch_size:
            pending["ppo_epoch"] += 1
            pending["permutation"] = None
            pending["mini_start"] = 0
        rollout_done = (pending["ppo_epoch"] >= args.ppo_update_iters or
                        (pending["stop_ppo"] and pending["permutation"] is None))
        metric = {"epoch": session.cursor["epoch"], "batch": session.cursor["next_batch"],
                  "ppo_epoch": pending["ppo_epoch"], "loss": float(combined.detach()),
                  "policy_loss": float(policy.detach()), "value_loss": float(value_loss.detach()),
                  "approx_kl": float(approx_kl.detach()), "kl_ref": float(kl_ref.detach()),
                  "early_stop_kl": pending["stop_ppo"], "reward": float(pending["rewards"].mean()),
                  "response_length": float(pending["mask"].sum(1).mean())}
        # Counter intentionally persists across rollouts, matching upstream's accumulation schedule.
        update_due = session.cursor["ppo_accum_counter"] % args.accumulation_steps == 0 or rollout_done
        if rollout_done:
            session.cursor["pending"] = None
        if update_due:
            checked_update([actor, critic], list(session.optimizers.values()), list(session.schedulers.values()), args.grad_clip)
            session.commit(metric)
            if session.limit_reached():
                return
        del values, logps, aux, policy, value_loss, approx_kl, kl_ref, loss, combined, pending


def _file_contract(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _validate_runtime(args, algorithm):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or args.rollout_engine != "torch" or args.use_compile:
        raise ValueError("Audited local runtime supports single-process torch rollout without compile only")
    if args.num_workers != 0:
        raise ValueError("Exact local RNG/cursor resume requires explicit --num_workers 0")
    if args.use_wandb:
        raise ValueError("This local runtime writes JSONL only; external tracking is disabled")
    if args.max_steps < 0 or min(args.epochs, args.batch_size, args.accumulation_steps, args.max_gen_len, args.save_interval) < 1:
        raise ValueError("Invalid positive training dimensions or negative --max_steps")
    if algorithm == "grpo" and args.num_generations < 2:
        raise ValueError("GRPO/CISPO requires complete groups with G >= 2")
    if algorithm == "grpo" and args.updates_per_rollout < 1:
        raise ValueError("--updates_per_rollout must be positive")
    if algorithm == "grpo" and args.updates_per_rollout > 1 and args.accumulation_steps != 1:
        raise ValueError("rollout-reuse requires --accumulation_steps 1 so every replay observes an updated actor")
    if algorithm == "ppo" and min(args.ppo_update_iters, args.mini_batch_size) < 1:
        raise ValueError("PPO update/minibatch dimensions must be positive")
    args.reward_device = args.reward_device or args.device
    args.resume_dir = str(Path(args.resume_dir or (Path(args.save_dir) / (args.save_weight + "_runtime"))).resolve())
    for path in (Path(args.save_dir).resolve(), Path(args.resume_dir)):
        if not path.is_relative_to(ROOT) or path == ROOT:
            raise ValueError(f"New outputs must be in a specific owned evomind subdirectory: {path}")
    if args.max_steps and Path(args.save_dir).resolve() == Path(args.init_dir).resolve():
        raise ValueError("Probe --save_dir must be separate from the initial/full-run weights directory")
    if args.save_weight == args.from_weight and Path(args.save_dir).resolve() == Path(args.init_dir).resolve():
        raise ValueError("Output would overwrite the initial/reference checkpoint")


def run_local(args, upstream, algorithm):
    """CLI entry; heavyweight models are loaded only when the user invokes training."""
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from dataset.lm_dataset import RLAIFDataset
    from trainer.rollout_engine import create_rollout_engine

    _validate_runtime(args, algorithm)
    seed_rng(args.seed, args.device)
    suffix = "_moe" if args.use_moe else ""
    initial_path = Path(args.init_dir).resolve() / f"{args.from_weight}_{args.hidden_size}{suffix}.pth"
    if not initial_path.is_file():
        raise FileNotFoundError(f"Required initial SFT weights missing: {initial_path}")
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    excluded = {"max_steps", "resume", "from_resume", "resume_dir", "save_dir", "debug_mode", "debug_interval", "debug_log_ratio", "log_interval", "save_interval"}
    contract = {"algorithm": algorithm, "settings": {key: value for key, value in vars(args).items() if key not in excluded},
                "initial": _file_contract(initial_path), "data": _file_contract(args.data_path),
                "tokenizer": {path.name: _file_contract(path) for path in sorted(Path(args.tokenizer_path).glob("*")) if path.is_file()},
                "reward_model": {path.name: _file_contract(path) for path in sorted(Path(args.reward_model_path).glob("*")) if path.is_file()},
                "runtime_source": _file_contract(__file__), "upstream_source": _file_contract(upstream.__file__),
                "torch_version": torch.__version__, "hardware_adaptation": {
                    "reward_device": args.reward_device, "reward_dtype": args.reward_dtype,
                    "policy_recomputation": "full_batch_upstream", "data_workers": 0,
                    "shuffle": "official seed+epoch randperm; direct single-process loading, no DataLoader base-seed draw"}}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    weights = torch.load(initial_path, map_location="cpu", weights_only=True)
    actor = MiniMindForCausalLM(config)
    actor.load_state_dict(weights, strict=True)
    actor = actor.to(args.device)
    reference = MiniMindForCausalLM(config)
    reference.load_state_dict(weights, strict=True)
    reference = reference.to(args.device).eval().requires_grad_(False)
    models = {"actor": actor}
    if algorithm == "ppo":
        critic = upstream.CriticModel(config)
        incompatible = critic.load_state_dict(weights, strict=False)
        if set(incompatible.missing_keys) != {"value_head.weight", "value_head.bias"} or incompatible.unexpected_keys:
            raise ValueError(f"Unexpected critic initialization keys: {incompatible}")
        models["critic"] = critic.to(args.device)
    del weights
    reward_model = LocalRewardModel(args.reward_model_path, device=args.reward_device, dtype=getattr(torch, args.reward_dtype))
    engine = create_rollout_engine("torch", actor, tokenizer, device=args.device, autocast_ctx=_context(args))
    # Upstream GRPO refers to config.max_seq_len, which is absent in this pinned
    # MiniMindConfig. RLAIFDataset only stores max_length and does not consume it;
    # actual prompt truncation remains the unchanged tokenizer/args.max_seq_len path.
    max_length = args.max_seq_len + args.max_gen_len
    dataset = RLAIFDataset(args.data_path, tokenizer, max_length=max_length, thinking_ratio=args.thinking_ratio)
    batches = math.ceil(len(dataset) / args.batch_size)
    if not batches:
        raise ValueError("Empty RL training dataset")
    optimizer_class = CPUOnlyAdamW if torch.device(args.device).type == "cpu" else torch.optim.AdamW
    optimizers = {"actor": optimizer_class(actor.parameters(), lr=args.learning_rate)}
    if algorithm == "ppo":
        optimizers["critic"] = optimizer_class(models["critic"].parameters(), lr=args.critic_learning_rate)
        total_steps = math.ceil(batches * args.epochs * args.ppo_update_iters * max(1, math.ceil(args.batch_size / args.mini_batch_size)) / args.accumulation_steps)
    else:
        total_steps = math.ceil(batches / args.accumulation_steps) * args.epochs * args.updates_per_rollout
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=optimizer.param_groups[0]["lr"] / 10)
        for name, optimizer in optimizers.items()}
    contract["planned_optimizer_steps"] = total_steps
    contract["dataset_rows"] = len(dataset)
    print(json.dumps({"event": "runtime_contract", "contract": contract}, ensure_ascii=False), flush=True)
    session = TrainingSession(args, algorithm, models, optimizers, schedulers, contract, total_batches=batches)
    engine.update_policy(actor)
    try:
        prepared = dict(actor=actor, reference=reference, tokenizer=tokenizer, dataset=dataset, engine=engine,
                        reward_fn=lambda prompts, responses: upstream.calculate_rewards(prompts, responses, reward_model))
        if algorithm == "ppo":
            run_ppo_prepared(session, critic=models["critic"], **prepared)
        else:
            run_grpo_prepared(session, **prepared)
        export = session.export_actor()
        reason = "epochs_complete" if session.exhausted() else "max_steps"
        result = session.write_summary("probe_complete" if args.max_steps else "complete", reason, final_checkpoint=export)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    except BaseException as exc:
        session.write_summary("failed", "exception", error=f"{type(exc).__name__}: {exc}")
        raise
