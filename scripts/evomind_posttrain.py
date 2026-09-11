"""Durable MiniMind-3 post-SFT branches, with probes distinct from full training.

This is reached by the EXISTING supervisor after SFT; never start a second GPU
trainer while that supervisor is training. Independent branch failures are saved
and do not prevent other branches from being attempted. They DO prevent project
acceptance and vision training. No implicit loss/data/G/length/epoch reductions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import traceback

from evomind_run import atomic_json, exclusive_run_lock, managed_child, now, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/runs/text_posttrain_20260908"
PLAN = ROOT / "configs/text_posttrain.json"
BASE_RUN = ROOT / "artifacts/runs/text_official_mini_20260908"


def build_plan():
    common = {"epochs": 1, "dtype": "bfloat16", "num_workers": 0,
              "grad_clip": 1.0, "use_compile": 0, "seed": 42}
    definitions = [
        ("dpo", "train_dpo.py", "offline", "dpo.jsonl", {
            "learning_rate": 4e-8, "max_seq_len": 1024, "beta": .15}, [(4, 1), (2, 2), (1, 4)]),
        ("lora", "train_lora.py", "offline", "lora_medical.jsonl", {
            "epochs": 10, "learning_rate": 1e-4, "max_seq_len": 340}, [(32, 1), (8, 4), (1, 32)]),
        ("grpo", "train_grpo.py", "rl", "rlaif.jsonl", {
            "learning_rate": 3e-7, "max_seq_len": 768, "max_gen_len": 1024,
            "num_generations": 6, "beta": .1, "epsilon": .2, "epsilon_high": 5.,
            "thinking_ratio": .9, "loss_type": "grpo"}, [(2, 1), (1, 2)]),
        ("cispo", "train_grpo.py", "rl", "rlaif.jsonl", {
            "learning_rate": 3e-7, "max_seq_len": 768, "max_gen_len": 1024,
            "num_generations": 6, "beta": .1, "epsilon": .2, "epsilon_high": 5.,
            "thinking_ratio": .9, "loss_type": "cispo"}, [(2, 1), (1, 2)]),
        ("agent_cispo", "train_agent.py", "agent", "agent_rl.jsonl", {
            "learning_rate": 3e-7, "max_seq_len": 1024, "max_gen_len": 768,
            "max_total_len": 2500, "max_turns": 3, "num_generations": 4,
            "beta": .1, "epsilon": .2, "epsilon_high": 5.,
            "thinking_ratio": .1, "loss_type": "cispo"}, [(2, 1), (1, 2)]),
        ("distillation", "train_distillation.py", "offline", "sft_t2t_mini.jsonl", {
            "epochs": 6, "learning_rate": 5e-6, "max_seq_len": 340, "alpha": .5,
            "temperature": 1.5, "student_hidden_size": 768, "student_num_layers": 8,
            "teacher_hidden_size": 768, "teacher_num_layers": 8, "student_use_moe": 0,
            "teacher_use_moe": 1, "from_student_weight": "full_sft",
            "from_teacher_weight": "full_sft", "teacher_autocast": 1,
            "teacher_init_dir": str(ROOT / "models/teacher_official")}, [(32, 1), (8, 4), (1, 32)]),
    ]
    branches = []
    for name, script, runtime, data, options, batches in definitions:
        flags = {**common, **options}
        if name != "distillation":
            flags.update(hidden_size=768, num_hidden_layers=8, use_moe=0, from_weight="full_sft")
        if runtime in ("rl", "agent"):
            flags.update(reward_model_path=str(ROOT / "models/internlm2-1_8b-reward"),
                         reward_device="cpu", reward_dtype="float32", rollout_engine="torch")
        branches.append({"name": name, "script": script, "runtime": runtime,
            "data": f"dataset/{data}", "options": flags,
            "batch_candidates": [{"batch_size": b, "accumulation_steps": a} for b, a in batches],
            "probe_optimizer_updates": 2,
            "adaptations": ["Windows workers=0; no compile", "OOM-only batch/accumulation fallback; no bitwise-equivalence claim"]
                + (["RM CPU float32 offload; unchanged upstream reward formula; numerical/runtime deviation"] if runtime != "offline" else [])
                + (["Public pinned MoE teacher; teacher autocast BF16; NOT a self-trained teacher"] if name == "distillation" else [])})
    return {"schema_version": 1, "upstream_commit": "6fc918beb68a0d8c40452338df6319fe168014ba",
            "base_run": str(BASE_RUN), "run_dir": str(RUN), "branches": branches,
            "weight_lineage": "All branches independently initialize from verified full_sft; not serial branch outputs",
            "probe_rule": "At least two complete updates on full source data; proof of integration only, not full length coverage or model quality",
            "acceptance_rule": "All configured full branches plus upstream question/tool tests and seven harness benchmarks; human review cancelled by user; pending is never success"}


def materialize():
    plan = build_plan()
    if PLAN.exists() and json.loads(PLAN.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing post-training plan differs; refusing to silently mutate a running contract")
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    if not PLAN.exists():
        atomic_json(PLAN, plan)
    return plan


def verify_record(record):
    path = Path(record["path"])
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"Missing or changed recorded artifact: {path}")
    if record.get("runtime_receipt"):
        receipt = Path(record["runtime_receipt"])
        if not receipt.is_file() or sha256(receipt) != record.get("runtime_receipt_sha256"):
            raise ValueError(f"Changed training receipt: {receipt}")
    if record.get("resume_checkpoint"):
        resume = Path(record["resume_checkpoint"])
        if not resume.is_file() or sha256(resume) != record.get("resume_checkpoint_sha256"):
            raise ValueError(f"Changed complete resume checkpoint: {resume}")
    return path


def base_checkpoint():
    state = json.loads((BASE_RUN / "state.json").read_text(encoding="utf-8"))
    for name in ("pretrain", "full_sft"):
        if state.get("stages", {}).get(name, {}).get("status") != "completed":
            raise RuntimeError(f"Post-training held: {name} has not completed")
    return verify_record(state["stages"]["full_sft"]["snapshot"])


def command(branch, candidate, directory, base, *, probe, resume=False):
    flags = {**branch["options"], **candidate, "data_path": str(ROOT / branch["data"]),
             "init_dir": str(base.parent), "save_dir": str(directory / "weights"),
             "resume_dir": str(directory / "checkpoints"), "max_steps": 2 if probe else 0,
             "log_interval": 1 if probe else 100, "save_interval": 1 if probe else 100,
             "from_resume": int(resume)}
    flags["lora_name" if branch["name"] == "lora" else "save_weight"] = branch["name"]
    argv = [str(ROOT / "trainer" / branch["script"])]
    if branch["runtime"] == "rl":
        argv.append("--evomind_runtime")
    for key, value in flags.items():
        argv.extend(["--" + key, str(value)])
    return argv


def observed_metric(line):
    if line.startswith("{"):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(value, dict) and "optimizer_update" in value:
            return {"scope": "observed_optimizer_update", **value}
        return None
    position = re.search(r"Epoch:\s*\[(\d+)/(\d+)\]\((\d+)/(\d+)\)", line)
    if not position:
        return None
    epoch, epochs, step, iters = map(int, position.groups())
    fields = {key: float(value) for key, value in re.findall(
        r"([A-Za-z_]+):\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", line)}
    return {"scope": "observed_training_microbatch_not_validation", "epoch": epoch,
            "epochs": epochs, "microstep": step, "global_microstep": (epoch-1)*iters+step, **fields}


def plot_observations(path):
    if not path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for index, keys in enumerate((("loss", "dpo_loss", "distill", "ce", "logits_loss", "policy_loss", "critic_loss"),
                                  ("reward", "reward_mean", "kl", "aux_loss", "lr", "learning_rate"))):
        for key in keys:
            points = [(i+1, row[key]) for i, row in enumerate(rows) if isinstance(row.get(key), (int, float))]
            if points:
                axes[index].plot(*zip(*points), label=key, linewidth=1)
        if axes[index].lines:
            axes[index].legend()
        axes[index].set_xlabel("Logged observation index (see JSONL for microstep/update and attempt)")
    fig.suptitle("Observed training metrics; no inferred validation performance")
    fig.tight_layout()
    target = path.with_name("curves.png")
    temporary = target.with_suffix(".png.tmp")
    fig.savefig(temporary, format="png", dpi=140)
    plt.close(fig)
    os.replace(temporary, target)


def execute(argv, directory):
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "console.log"
    metrics_path = directory / "observed.metrics.jsonl"
    attempt = now()
    count = 0
    environment = os.environ.copy()
    environment.update(HF_HOME=str(ROOT / ".cache/huggingface"),
        HF_DATASETS_CACHE=str(ROOT / ".cache/huggingface/datasets"),
        HF_MODULES_CACHE=str(ROOT / ".cache/huggingface/modules"),
        TORCH_HOME=str(ROOT / ".cache/torch"), TOKENIZERS_PARALLELISM="false",
        PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    with log.open("a", encoding="utf-8") as output:
        output.write(f"\nSTART {now()} {json.dumps(argv)}\n")
        output.flush()
        with managed_child([sys.executable, "-B", "-u", *argv], cwd=ROOT / "trainer",
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           encoding="utf-8", errors="replace", bufsize=1, env=environment) as process:
            atomic_json(directory / "launch.json", {"pid": process.pid, "started_at": now(), "argv": argv})
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)
                metric = observed_metric(line)
                if metric:
                    with metrics_path.open("a", encoding="utf-8") as metrics:
                        metrics.write(json.dumps({"attempt_started_at": attempt, **metric}, allow_nan=False)+"\n")
                    count += 1
                    if count % 100 == 0:
                        try:
                            plot_observations(metrics_path)
                        except Exception as error:
                            output.write(f"CURVE_ERROR {error}\n")
            code = process.wait()
    try:
        plot_observations(metrics_path)
    except Exception as error:
        atomic_json(directory / "curve_error.json", {"error": str(error), "metrics_preserved": str(metrics_path)})
    return code, log


def receipt_path(branch, directory):
    return (directory / "checkpoints/summary.json" if branch["runtime"] == "rl" else
            directory / "weights" / f"{branch['name']}_768.runtime.json")


def check_receipt(branch, directory, *, probe):
    """Validate native runtime receipts without importing torch or loading pickle.

    Tensor-state integrity belongs to each strict runtime loader. Here the
    completed receipt is bound to its plan, cursor, contract and actual files;
    returned hashes make that CPU-only acceptance evidence auditable.
    """
    directory = Path(directory).resolve()
    path = receipt_path(branch, directory)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or type(data.get("probe_only")) is not bool or data["probe_only"] != probe:
        raise ValueError("Probe/full receipt mismatch")

    def integer(mapping, key, minimum=0):
        value = mapping.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid or missing receipt integer: {key}")
        return value

    def bound_file(value, expected):
        if not isinstance(value, str) or not value or Path(value).resolve() != expected.resolve():
            raise ValueError(f"Receipt file path differs from expected artifact: {expected}")
        if not expected.is_file() or not expected.stat().st_size:
            raise FileNotFoundError(f"Missing nonempty runtime artifact: {expected}")
        return sha256(expected)

    def checked_hash(value, actual, label):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) or value != actual:
            raise ValueError(f"Missing or changed {label} SHA256")

    def contract_hash(contract):
        if not isinstance(contract, dict) or not contract:
            raise ValueError("Missing training contract")
        return hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def code_hashes(mapping):
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Missing runtime code hashes")
        for filename, digest in mapping.items():
            source = Path(filename)
            if not source.is_file():
                raise FileNotFoundError(f"Missing recorded implementation: {source}")
            checked_hash(digest, sha256(source), "implementation")

    runtime = branch["runtime"]
    total = integer(data, "optimizer_updates", 1)
    invocation_key = "invocation_optimizer_updates" if runtime == "rl" else "optimizer_updates_this_invocation"
    updates = integer(data, invocation_key)
    if updates > total:
        raise ValueError("Invocation updates exceed cumulative updates")
    requested_steps = branch["probe_optimizer_updates"] if probe else 0
    if probe:
        if data.get("status") != "probe_complete" or updates < max(2, requested_steps) or data.get("stop_reason") != "max_steps":
            raise ValueError("Probe did not finish at least two optimizer updates")
    elif data.get("status") not in ("completed", "complete") or data.get("stop_reason") != "epochs_complete":
        raise ValueError("Trainer did not complete full configured epochs")

    epochs = branch["options"]["epochs"]
    export = directory / "weights" / f"{branch['name']}_768.pth"
    resume = (directory / "checkpoints/latest.resume.pt" if runtime == "rl" else
              directory / "checkpoints" / f"{branch['name']}_768_resume.pth")
    if runtime == "rl":
        algorithm = "ppo" if branch["name"] == "ppo" else "grpo"
        if (data.get("schema_version") != "evomind.rl.run/1" or data.get("algorithm") != algorithm
                or data.get("loss_type") != branch["options"].get("loss_type")):
            raise ValueError("RL algorithm/loss/schema differs from branch plan")
        config = json.loads((directory / "checkpoints/run_config.json").read_text(encoding="utf-8"))
        contract = config.get("contract")
        digest = contract_hash(contract)
        checked_hash(data.get("contract_sha256"), digest, "contract")
        if config.get("algorithm") != algorithm or contract.get("algorithm") != algorithm:
            raise ValueError("RL run configuration algorithm differs")
        params = contract.get("settings")
        invocation = config.get("invocation")
        if not isinstance(params, dict) or not isinstance(invocation, dict):
            raise ValueError("Missing RL settings/invocation")
        if any(key not in invocation or invocation[key] != value for key, value in params.items()):
            raise ValueError("RL invocation differs from hashed settings")
        if integer(invocation, "max_steps") != requested_steps or integer(data, "requested_max_steps") != requested_steps:
            raise ValueError("RL probe/full update limit differs from invocation")
        for key, expected in (("save_dir", export.parent), ("resume_dir", resume.parent)):
            if not isinstance(invocation.get(key), str) or Path(invocation[key]).resolve() != expected:
                raise ValueError("RL invocation output directory differs")
        if integer(data, "configured_epochs", 1) != epochs:
            raise ValueError("RL epoch budget differs from branch plan")
        planned = integer(contract, "planned_optimizer_steps", 1)
        if integer(data, "planned_optimizer_updates", 1) != planned:
            raise ValueError("RL planned updates differ from hashed contract")
        integer(contract, "dataset_rows", 1)
        if total > planned:
            raise ValueError("RL cumulative updates exceed the nominal budget")
        if integer(data, "checkpoint_optimizer_update", 1) != total:
            raise ValueError("RL final update is not committed to a resume checkpoint")
        if integer(data, "resume_start_optimizer_update") + updates != total:
            raise ValueError("RL invocation/resume update accounting differs")
        if type(data.get("pending_ppo_rollout")) is not bool:
            raise ValueError("RL receipt lacks explicit pending-rollout state")
        cursor_epoch = integer(data, "cursor_epoch")
        integer(data, "cursor_next_batch")
        if cursor_epoch > epochs or (not probe and (cursor_epoch != epochs or data["pending_ppo_rollout"]
                                                   or data["cursor_next_batch"] != 0)):
            raise ValueError("RL/PPO cursor has not exhausted complete epochs")
        # PPO KL early-stop can legitimately make actual updates < planned.
        for key in ("initial", "data", "runtime_source", "upstream_source"):
            record = contract.get(key)
            if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))):
                raise ValueError(f"Missing RL input/source contract: {key}")
        code_hashes({contract[key]["path"]: contract[key]["sha256"] for key in ("runtime_source", "upstream_source")})
        for key in ("tokenizer", "reward_model", "hardware_adaptation"):
            if not isinstance(contract.get(key), dict) or not contract[key]:
                raise ValueError(f"Missing RL contract component: {key}")
        export_hash = bound_file(data.get("final_checkpoint"), export)
        resume_hash = bound_file(data.get("resume_checkpoint"), resume)
    else:
        contract = data.get("contract")
        digest = contract_hash(contract)
        params = contract.get("parameters") if runtime == "offline" else contract
        if not isinstance(params, dict):
            raise ValueError("Missing Offline/Agent parameters")
        if params.get("epochs") != epochs:
            raise ValueError("Offline/Agent epoch budget differs from branch plan")
        if runtime == "offline":
            if contract.get("version") != 1 or contract.get("branch") != branch["name"]:
                raise ValueError("Offline contract branch/schema differs")
            if data.get("optimizer_boundary") is not True or integer(params, "max_steps") != requested_steps:
                raise ValueError("Offline receipt is not the requested optimizer boundary")
            code_hashes(contract.get("code_sha256"))
            code_hashes(data.get("code_sha256"))
            export_hash = bound_file(data.get("export_path"), export)
            resume_hash = bound_file(data.get("resume_path"), resume)
            iters = integer(data, "iters", 1)
        else:
            if data.get("format") != "evomind.agent.optimizer-boundary/1" or contract.get("probe_only") is not probe:
                raise ValueError("Agent format/probe contract differs")
            if integer(data, "pending_microsteps") != 0:
                raise ValueError("Agent receipt contains uncommitted microsteps")
            if integer(data, "invocation_updates") != updates:
                raise ValueError("Agent invocation counters differ")
            code_hashes({str(ROOT / name): value for name, value in contract.get("implementation_sha256", {}).items()})
            export_hash = bound_file(data.get("export"), export)
            resume_hash = bound_file(data.get("resume"), resume)
            iters = math.ceil(integer(params, "dataset_rows", 1) / integer(params, "batch_size", 1))
        checked_hash(data.get("export_sha256"), export_hash, "export")
        checked_hash(data.get("resume_sha256"), resume_hash, "resume")
        epoch, step = integer(data, "epoch"), integer(data, "step", 1)
        if epoch >= epochs or step > iters:
            raise ValueError("Offline/Agent cursor exceeds configured budget")
        if not probe:
            if epoch + 1 != epochs or step != iters:
                raise ValueError("Offline/Agent cursor is not at final epoch tail")

    # Agent's native contract deliberately omits CLI-only fields. All recorded
    # objective/model/budget fields, and every Offline/RL option, remain required.
    agent_omitted = {"num_workers", "use_compile", "from_weight", "rollout_engine"}
    for key, expected in branch["options"].items():
        if runtime == "agent" and key in agent_omitted:
            continue
        if key not in params or params[key] != expected:
            raise ValueError(f"Training contract differs from branch option: {key}")
    candidate = {key: integer(params, key, 1) for key in ("batch_size", "accumulation_steps")}
    if candidate not in branch["batch_candidates"]:
        raise ValueError("Training batch/accumulation is not an authorized candidate")
    if runtime != "agent" and params.get("lora_name" if branch["name"] == "lora" else "save_weight") != branch["name"]:
        raise ValueError("Training weight prefix differs from branch identity")
    return {"path": str(export), "sha256": export_hash, "runtime_receipt": str(path),
            "runtime_receipt_sha256": sha256(path), "probe_only": probe,
            "resume_checkpoint": str(resume), "resume_checkpoint_sha256": resume_hash,
            "contract_sha256": digest,
            "verification_scope": "CPU JSON/contracts/file hashes; tensor states are validated by the native resume loader"}


def validate_assets(branch):
    report = json.loads((ROOT / "artifacts/provenance/posttrain_assets.json").read_text(encoding="utf-8"))
    if report.get("status") != "files_verified":
        raise RuntimeError("Pinned post-training files are not verified")
    data_path = ROOT / branch["data"]
    if branch["name"] != "distillation":
        item = next(r for r in report["assets"] if Path(r["path"]) == data_path)
        verify_record(item)
    if branch["name"] == "distillation":
        text_assets = json.loads((ROOT / "artifacts/provenance/text_assets.json").read_text(encoding="utf-8"))
        source_item = next(r for r in text_assets["assets"] if r["filename"] == "sft_t2t_mini.jsonl")
        verify_record({"path": str(data_path), "sha256": source_item["sha256"]})
    if branch["runtime"] in ("rl", "agent"):
        review_path = ROOT / "artifacts/provenance/reward_code_review.json"
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("status") != "reviewed":
            raise RuntimeError("Reward custom-code review is pending")
        for item in review["files"]:
            verify_record(item)
    if branch["name"] == "distillation":
        verify_teacher_mapping()


def verify_teacher_mapping():
    """Check full tokenization behavior JSON (metadata formatting may differ)."""
    source = ROOT / "models/teacher_official/tokenizer/tokenizer.json"
    local = ROOT / "model/tokenizer.json"
    if json.loads(source.read_text(encoding="utf-8")) != json.loads(local.read_text(encoding="utf-8")):
        raise ValueError("Teacher/local tokenization differs; weight shape alone cannot establish token-ID compatibility")
    report = json.loads((ROOT / "artifacts/provenance/posttrain_assets.json").read_text(encoding="utf-8"))
    item = next(r for r in report["assets"] if r["filename"] == "full_sft_768_moe.pth")
    verify_record(item)  # trainer additionally enforces strict architecture and logits vocabulary
    tokenizer_item = next(r for r in report["assets"] if r["filename"] == "tokenizer.json")
    verify_record(tokenizer_item)


def run_branch(branch, state, persist, base, execute_fn=execute):
    name = branch["name"]
    row = state["branches"].setdefault(name, {"status": "pending", "probes": []})
    if row.get("remote_dispatch") and row.get("status") != "completed":
        raise RuntimeError("Stage is assigned to a remote supervisor; verify/import that run before any local restart")
    if row.get("status") == "completed":
        verify_record(row["output"])
        return
    validate_assets(branch)
    directory = RUN / name
    candidate = row.get("selected")
    if candidate is not None:
        passed = next(r for r in reversed(row["probes"]) if r["candidate"] == candidate and r["status"] == "passed")
        verify_record(passed["output"])
        check_receipt(branch, Path(passed["directory"]), probe=True)
    if candidate is None:
        for index, item in enumerate(branch["batch_candidates"]):
            probe_dir = directory / f"probe_{index}"
            previous = next((r for r in row["probes"] if r["candidate"] == item), None)
            if previous and previous["status"] == "oom":
                continue
            if previous and previous["status"] == "passed":
                check_receipt(branch, Path(previous["directory"]), probe=True)
                candidate = item
                break
            if probe_dir.exists():
                # Preserve an interrupted probe and rerun from the base in a new owned attempt.
                attempt = 1
                while (directory / f"probe_{index}_retry{attempt}").exists():
                    attempt += 1
                probe_dir = directory / f"probe_{index}_retry{attempt}"
            row.update(status="probing", current_probe=str(probe_dir))
            persist()
            code, log = execute_fn(command(branch, item, probe_dir, base, probe=True), probe_dir)
            if code:
                is_oom = "out of memory" in log.read_text(encoding="utf-8").lower()
                row["probes"].append({"candidate": item, "status": "oom" if is_oom else "failed", "log": str(log)})
                persist()
                if is_oom:
                    continue
                raise RuntimeError(f"{name} probe failed (not OOM); inspect {log}")
            receipt = check_receipt(branch, probe_dir, probe=True)
            row["probes"].append({"candidate": item, "status": "passed", "directory": str(probe_dir), "output": receipt})
            candidate = item
            break
        if candidate is None:
            raise RuntimeError(f"All audited batch candidates OOM for {name}; no silent G/length/epoch reduction")
        row["selected"] = candidate
        persist()
    full = Path(row.get("full_directory", directory / "full"))
    if full.resolve().parent != directory.resolve():
        raise ValueError("Recorded full-run directory is outside this branch")
    def resume_file(path):
        return (path / "checkpoints/latest.resume.pt" if branch["runtime"] == "rl" else
                path / "checkpoints" / f"{name}_768_resume.pth")
    resume_path = resume_file(full)
    if full.exists() and any(full.iterdir()) and not resume_path.exists():
        # A failure before the first save has logs/config, but cannot be resumed.
        # Preserve that evidence and restart FROM SFT in a fresh full attempt.
        attempt = 1
        while (directory / f"full_retry{attempt}").exists():
            attempt += 1
        row.setdefault("abandoned_full_attempts", []).append({"directory": str(full),
            "reason": "No complete resume checkpoint; fresh full restart from SFT, previous evidence retained",
            "recorded_at": now()})
        full = directory / f"full_retry{attempt}"
        resume_path = resume_file(full)
    row.update(status="training", full_directory=str(full), started_at=now())
    persist()
    code, log = execute_fn(command(branch, candidate, full, base, probe=False, resume=resume_path.exists()), full)
    if code:
        raise RuntimeError(f"{name} full training exited {code}; preserved resumable state and {log}")
    output = check_receipt(branch, full, probe=False)
    row.update(status="completed", output=output, completed_at=now())
    persist()


def run_posttraining(resume=False):
    if (ROOT / "configs/product_pipeline.json").is_file():
        raise RuntimeError("Historical independent branch queue superseded; use scripts/evomind_continue.py for the product route")
    plan = materialize()
    base = base_checkpoint()
    RUN.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(RUN):
        path = RUN / "state.json"
        plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if not resume:
                raise RuntimeError("Existing post-training state requires --resume")
            if state["plan_sha256"] != plan_hash or state["base"]["sha256"] != sha256(base):
                raise ValueError("Plan or SFT base changed across continuation")
        else:
            state = {"schema_version": 1, "plan_sha256": plan_hash, "started_at": now(),
                     "base": {"path": str(base), "sha256": sha256(base)}, "branches": {}}
        def persist():
            state["updated_at"] = now()
            atomic_json(path, state)
        state.update(status="running", pid=os.getpid())
        persist()
        for branch in plan["branches"]:
            try:
                run_branch(branch, state, persist, base)
            except Exception as error:
                state["branches"].setdefault(branch["name"], {}).update(status="failed", error=str(error), traceback=traceback.format_exc())
                persist()
                print(f"BRANCH_FAILED {branch['name']}: {error}; continuing independent branches", flush=True)
        failed = [b["name"] for b in plan["branches"] if state["branches"][b["name"]]["status"] != "completed"]
        state.update(status="training_branches_complete" if not failed else "incomplete", failed_branches=failed)
        persist()
        return 0 if not failed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        plan = materialize()
        print(f"Post-SFT: {len(plan['branches'])} independent full-epoch branches; no GPU launch")
    else:
        raise SystemExit(run_posttraining(args.resume))


if __name__ == "__main__":
    main()
