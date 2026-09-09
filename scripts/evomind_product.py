"""Evidence-gated product lineage. CPU orchestration; never duplicate a live run.

Reuse native training receipts and upstream-family evaluators. A failed job is
not a rejected candidate: it remains incomplete. Promotion is a conservative
engineering heuristic, not proof of statistical significance or useful video AI.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path

import evomind_posttrain as training
import evomind_posttrain_evaluate as evaluation
from evomind_harness_eval import TASKS
from evomind_run import atomic_json, exclusive_run_lock, now, sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/product_pipeline.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def settings():
    return read(CONFIG)


def metric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Missing/nonfinite/out-of-range quality metric")
    return float(value)


def benchmark_score(summary, task):
    # Use the native group aggregate for C-Eval/CMMLU, never average subjects
    # ourselves or substitute acc_norm for acc across checkpoints.
    scores = summary["scores"][task]
    return metric(scores["acc,none"])


def compare(base, candidate, policy):
    if set(base["benchmarks"]) != set(TASKS) or set(candidate["benchmarks"]) != set(TASKS):
        raise ValueError("All seven native benchmark aggregates are required")
    deltas = {task: metric(candidate["benchmarks"][task]) - metric(base["benchmarks"][task]) for task in TASKS}
    eos_delta = metric(candidate["eos"]) - metric(base["eos"])
    repeat_delta = metric(candidate["repeat4"]) - metric(base["repeat4"])
    safe = (min(deltas.values()) >= -policy["benchmark_max_drop"]
            and eos_delta >= -policy["eos_max_drop"] and repeat_delta <= policy["repeat_max_increase"])
    improved = (max(deltas.values()) >= policy["minimum_improvement"]
                or eos_delta >= policy["minimum_improvement"] or -repeat_delta >= policy["minimum_improvement"])
    return {"promote": safe and improved, "benchmark_deltas": deltas,
            "eos_delta": eos_delta, "repeat4_delta": repeat_delta,
            "reason": "nonregression_and_improvement" if safe and improved else "retain_previous_checkpoint",
            "claim": "Engineering selection only; not statistical significance or video quality evidence"}


def baseline_ready(scores, policy):
    return metric(scores["eos"]) >= policy["baseline_min_eos"] and metric(scores["repeat4"]) <= policy["baseline_max_repeat4"]


def model(record):
    return {"checkpoint": str(training.verify_record(record)), "sha256": record["sha256"]}


def evaluate(models):
    code = evaluation.run_evaluations(models_override=models)
    if code:
        raise RuntimeError("Evaluation incomplete; no model promotion or downstream training")
    state = read(evaluation.EVAL / "state.json")
    scores = {}
    evidence = []
    for name, item in models.items():
        reports = {}
        for key, _, _, result in evaluation.jobs(name, item):
            row = state["models"][name][key]
            if row.get("status") != "completed" or row.get("inputs") != item:
                raise ValueError("Evaluation lineage mismatch")
            training.verify_record(row["result"])
            reports[key] = evaluation.verify_evaluation_result(key, result, item)
            evidence.append(row["result"])
        records = read_lines(evaluation.EVAL / name / "thinking_off/records.jsonl")
        if len(records) != 8:
            raise ValueError("Expected eight official non-thinking examples")
        # Avoid depending on display-summary aggregation schemas.
        scores[name] = {"benchmarks": {task: benchmark_score(reports[f"harness_{task}"], task) for task in TASKS},
                        "eos": sum(bool(r["eos"]) for r in records) / len(records),
                        "repeat4": sum(metric(r["repeat_4"]) for r in records) / len(records)}
    return scores, evidence


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def configure_paths(plan):
    training.RUN = ROOT / plan["run_dir"]
    evaluation.RUN = training.RUN
    evaluation.EVAL = ROOT / plan["eval_dir"]
    return training.RUN


def selected_checkpoint():
    plan = settings()
    run = ROOT / plan["run_dir"]
    receipt = read(run / "product_acceptance.json")
    if receipt.get("status") != "accepted_text_screening" or receipt.get("config_sha256") != sha256(CONFIG):
        raise RuntimeError("Text product selection not accepted under the current contract")
    state = read(run / "state.json")
    if receipt.get("state_sha256") != sha256(run / "state.json"):
        raise ValueError("Training/selection state changed after acceptance")
    candidates = {"full_sft": state["base"]}
    for branch in plan["stages"]:
        row = state["branches"][branch["name"]]
        if row.get("status") != "completed":
            raise RuntimeError("Required training stage incomplete")
        training.verify_record(row["initial"])
        training.verify_record(row["output"])
        candidates[branch["name"]] = row["output"]
    selected = receipt["selected_name"]
    if receipt["selected"] != candidates[selected] or state["selected_name"] != selected:
        raise ValueError("Selected checkpoint is not the accepted training artifact")
    for record in receipt["evaluation_evidence"]:
        training.verify_record(record)
    # Recheck native contracts and raw prediction hashes; a summary alone is not enough.
    previous_eval = evaluation.EVAL
    try:
        evaluation.EVAL = ROOT / plan["eval_dir"]
        for name, record in candidates.items():
            item = model(record)
            for key, _, _, result in evaluation.jobs(name, item):
                evaluation.verify_evaluation_result(key, result, item)
    finally:
        evaluation.EVAL = previous_eval
    return training.verify_record(receipt["selected"])


def run_product():
    plan = settings()
    run = configure_paths(plan)
    run.mkdir(parents=True, exist_ok=True)
    base = training.base_checkpoint()
    with exclusive_run_lock(run):
        path = run / "state.json"
        base_record = {"path": str(base), "sha256": sha256(base)}
        state = read(path) if path.exists() else {"config_sha256": sha256(CONFIG), "base": base_record,
            "branches": {}, "selections": {}, "selected_name": "full_sft", "started_at": now()}
        if state["config_sha256"] != sha256(CONFIG) or state["base"] != base_record:
            raise ValueError("Product plan or SFT base changed; preserve existing run")
        def persist():
            state["updated_at"] = now()
            atomic_json(path, state)
        try:
            state["status"] = "running"
            persist()
            candidates = {"full_sft": base_record}
            scores, evidence = evaluate({"full_sft": model(base_record)})
            if not baseline_ready(scores["full_sft"], plan["selection"]):
                state.update(status="needs_sft_repair", baseline_screening=scores["full_sft"])
                persist()
                raise RuntimeError("SFT fails explicit EOS/repetition screening; do not hide it with extra RL")
            selected = "full_sft"
            for recipe in plan["stages"]:
                name = recipe["name"]
                branch = copy.deepcopy(recipe)
                initial = candidates[selected]
                branch["options"]["from_weight"] = Path(initial["path"]).stem.removesuffix("_768")
                row = state["branches"].setdefault(name, {"status": "pending", "probes": []})
                if "initial" in row and row["initial"] != initial:
                    raise ValueError("Resume cannot change a stage's selected parent")
                row.update(initial=initial, initial_name=selected, effective_recipe=branch)
                persist()
                training.run_branch(branch, state, persist, training.verify_record(initial))
                candidates[name] = state["branches"][name]["output"]
                scores, evidence = evaluate({n: model(r) for n, r in candidates.items()})
                decision = compare(scores[selected], scores[name], plan["selection"])
                decision.update(parent=selected, candidate=name, parent_record=initial,
                                candidate_record=candidates[name], scores=scores, evaluation_evidence=evidence)
                if decision["promote"]:
                    selected = name
                decision["selected_name"] = selected
                state["selections"][name] = decision
                state["selected_name"] = selected
                persist()
            state.update(status="text_selected", selected_name=selected, completed_at=now())
            persist()
            atomic_json(run / "product_acceptance.json", {"status": "accepted_text_screening",
                "config_sha256": sha256(CONFIG), "state_sha256": sha256(path),
                "selected_name": selected, "selected": candidates[selected], "evaluation_evidence": evidence,
                "human_review": "cancelled_by_user", "video_quality": "not_evaluated",
                "agent": "deferred_nonblocking_extension", "selection_policy": plan["selection"]})
            return selected_checkpoint()
        except Exception as error:
            if state.get("status") != "needs_sft_repair":
                state["status"] = "incomplete"
            state["error"] = str(error)
            persist()
            raise


def enable_vision():
    selected = selected_checkpoint()
    plan = settings()
    receipt = ROOT / plan["run_dir"] / "product_acceptance.json"
    path = ROOT / "configs/text_alignment_scope.json"
    scope = read(path)
    scope.update(vision_enabled=True, text_acceptance=str(receipt), text_acceptance_sha256=sha256(receipt),
                 selected_text_checkpoint={"path": str(selected), "sha256": sha256(selected)})
    atomic_json(path, scope)
    return selected
