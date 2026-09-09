"""Evaluate available completed text branches using upstream task families.

No reward-model score is used as a replacement for human annotation. An unfinished
branch or benchmark remains pending and cannot open the vision training gate.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import hmac
import json
from pathlib import Path
import random
import secrets
import traceback

from evomind_harness_eval import TASKS
from evomind_run import atomic_json, now, sha256, exclusive_run_lock
from evomind_posttrain import ROOT, RUN, BASE_RUN, base_checkpoint, build_plan, execute, verify_record

EVAL = ROOT / "artifacts/evaluation/text_posttrain_20260908"


def model_inputs():
    base = base_checkpoint()
    models = {"full_sft": {"checkpoint": str(base), "sha256": sha256(base)}}
    state = json.loads((RUN / "state.json").read_text(encoding="utf-8"))
    for name, branch in state["branches"].items():
        if branch.get("status") != "completed":
            continue
        path = verify_record(branch["output"])
        models[name] = {"checkpoint": str(base if name == "lora" else path),
                        "sha256": sha256(base if name == "lora" else path)}
        if name == "lora":
            models[name].update(lora=str(path), lora_sha256=sha256(path))
    return models


def jobs(name, model):
    common = ["--checkpoint", model["checkpoint"]]
    if model.get("lora"):
        common += ["--lora", model["lora"]]
    tasks = []
    for thinking in (False, True):
        key = "thinking_on" if thinking else "thinking_off"
        dest = EVAL / name / key
        argv = [str(ROOT / "scripts/evomind_text_eval.py"), *common, "--output-dir", str(dest), "--resume"]
        if thinking:
            argv += ["--thinking"]
        tasks.append((key, argv, dest, dest / "summary.json"))
    for task in TASKS:
        key = f"harness_{task}"
        # New immutable receipt namespace; preserve failed v1 logs/contracts.
        dest = EVAL / name / (key + "_v2")
        tasks.append((key, [str(ROOT / "scripts/evomind_harness_eval.py"), *common,
                           "--task", task, "--output-dir", str(dest)], dest, dest / "summary.json"))
    path = Path(model["checkpoint"])
    for seed in (42,):
        key = f"tool_seed{seed}"
        dest = EVAL / name / key
        argv = [str(ROOT / "scripts/eval_toolcall.py"), "--backend", "local", "--auto",
                "--load_from", str(ROOT / "model"), "--save_dir", str(path.parent),
                "--weight", path.stem.removesuffix("_768"), "--seed", str(seed),
                "--max_turns", "3", "--max_new_tokens", "512", "--output_json", str(dest / "results.json")]
        if model.get("lora"):
            argv.extend(["--lora", model["lora"]])
        tasks.append((key, argv, dest, dest / "results.json"))
    return tasks


def verify_evaluation_result(key, result, model):
    if key.startswith("harness_"):
        from evomind_harness_eval import contract, verify_result
        return verify_result(result, contract(model["checkpoint"], model.get("lora"), key.removeprefix("harness_")))
    report = json.loads(result.read_text(encoding="utf-8"))
    if report.get("status") not in ("complete", "completed"):
        raise ValueError("Incomplete evaluation receipt")
    if key.startswith("tool_"):
        if (report.get("seed") != int(key.removeprefix("tool_seed")) or report.get("max_turns") != 3
                or report.get("max_new_tokens") != 512 or report.get("backend") != "local"
                or report.get("temperature") != .9 or report.get("top_p") != .9 or len(report.get("cases", [])) != 8):
            raise ValueError("Tool evaluation protocol differs")
        provenance = report["provenance"]
        if provenance["checkpoint"]["sha256"] != model["sha256"] or provenance.get("lora", {}).get("sha256") != model.get("lora_sha256"):
            raise ValueError("Tool checkpoint/adapter differs")
        for item in provenance.values():
            verify_record(item)
    else:
        if report.get("checkpoint_sha256") != model["sha256"] or report.get("lora_sha256") != model.get("lora_sha256"):
            raise ValueError("Diagnostic/benchmark checkpoint or adapter differs")
        if key.startswith("thinking_"):
            from evomind_text_eval import PROMPTS
            if report.get("thinking") != (key == "thinking_on") or report.get("seeds") != [42]:
                raise ValueError("Diagnostic thinking/seeds differ")
            if (report.get("evaluator_sha256") != sha256(ROOT / "scripts/evomind_text_eval.py")
                    or report.get("upstream_eval_sha256") != sha256(ROOT / "eval_llm.py")
                    or report.get("loader_sha256") != sha256(ROOT / "scripts/evomind_load_text_model.py")
                    or report.get("tokenizer_sha256") != {p.name: sha256(p) for p in sorted((ROOT / "model").glob("*.json"))}
                    or report.get("prompts") != json.loads(json.dumps(PROMPTS, ensure_ascii=False))
                    or report.get("max_new_tokens") != 8192
                    or report.get("sampling") != {"temperature": .85, "top_k": 50, "top_p": .95}):
                raise ValueError("Diagnostic source/prompt/tokenizer/decoding protocol changed")
            verify_record({"path": str(result.parent / "records.jsonl"), "sha256": report["records_sha256"]})
        elif key == "chinese":
            import evomind_chinese_eval as chinese
            argv = ["--checkpoint", model["checkpoint"], "--output-dir", str(result.parent), "--resume"]
            if model.get("lora"):
                argv += ["--lora", model["lora"]]
            args = chinese.parser().parse_args(argv)
            manifest = chinese.validate_prepared(args.data_dir)
            config = chinese.make_config(args, manifest)
            config_hash = chinese.digest(config)
            if chinese.read_json(result.parent / "config.json") != {"config_sha256": config_hash, "config": config}:
                raise ValueError("Chinese benchmark source/code/tokenizer/protocol changed")
            expected = {name: chinese.dataset_report(name, details, args.data_dir, result.parent, config_hash)
                        for name, details in manifest["datasets"].items()}
            if (report.get("config_sha256") != config_hash or report.get("datasets") != expected
                    or report.get("protocol") != chinese.PROTOCOL or report.get("limitations") != chinese.LIMITATIONS):
                raise ValueError("Chinese full-split prediction hashes/coverage/scores differ")
    return report


def write_blind_package(models, state):
    """Three user-requested prompts, 3 decoding seeds, each branch vs SFT.

    IDs hide model identity; two real reviewers per pair. This is a diagnostic
    blind review, not a representative population score. Key is separate.
    """
    directory = EVAL / "blind_review"
    directory.mkdir(parents=True, exist_ok=True)
    packages = []
    key = {}
    private_seed_path = directory / "identity_seed_PRIVATE.json"
    if not private_seed_path.exists():
        atomic_json(private_seed_path, {"salt": secrets.token_hex(32)})
    salt = bytes.fromhex(json.loads(private_seed_path.read_text(encoding="utf-8"))["salt"])
    base_file = EVAL / "full_sft/thinking_off/records.jsonl"
    if not base_file.exists():
        return {"status": "pending_baseline"}
    def selected(path):
        summary = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
        verify_record({"path": str(path), "sha256": summary["records_sha256"]})
        rows = [r for r in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
                if r["mode"] == "sampled" and r["prompt_id"] in (0, 1, 2)]
        identities = {(r["seed"], r["prompt_id"]): r for r in rows}
        if len(rows) != 9 or set(identities) != {(seed, prompt) for seed in (42, 123, 2026) for prompt in (0, 1, 2)}:
            raise ValueError("Blind package requires exactly 9 distinct matched diagnostic identities per model")
        return identities
    baseline = selected(base_file)
    for name in sorted(set(models) - {"full_sft"}):
        path = EVAL / name / "thinking_off/records.jsonl"
        if state["models"].get(name, {}).get("thinking_off", {}).get("status") != "completed":
            continue
        for identity, answer in sorted(selected(path).items()):
            if identity not in baseline:
                raise ValueError("Missing matched baseline answer")
            if answer["prompt"] != baseline[identity]["prompt"]:
                raise ValueError("Blind pair has different prompts")
            pair_id = hmac.new(salt, f"{name}:{identity}".encode(), hashlib.sha256).hexdigest()[:16]
            order = [("full_sft", baseline[identity]), (name, answer)]
            random.Random(pair_id).shuffle(order)
            packages.append({"pair_id": pair_id, "prompt": answer["prompt"],
                             "answer_A": order[0][1]["answer"], "answer_B": order[1][1]["answer"]})
            key[pair_id] = {"A": order[0][0], "B": order[1][0], "seed": identity[0], "prompt_id": identity[1],
                           "branch_records_sha256": sha256(path), "baseline_records_sha256": sha256(base_file)}
    packages.sort(key=lambda p: p["pair_id"])
    content = {"status": "awaiting_human_annotations", "required_reviewers_per_pair": 2,
               "scope": "3 fixed diagnostic prompts x 3 seeds; not broad human preference benchmark",
               "pairs": packages}
    if len(packages) != 9 * (len(models)-1):
        raise ValueError("Blind package is missing a model or matched pair")
    task_path = directory / "tasks.json"
    if task_path.exists() and json.loads(task_path.read_text(encoding="utf-8")) != content:
        raise ValueError("Blind tasks changed; preserve existing review instead of overwriting identities")
    atomic_json(task_path, content)
    atomic_json(directory / "identity_key_DO_NOT_GIVE_REVIEWERS.json", key)
    csv_path = directory / "annotations.csv"
    if not csv_path.exists():
        with csv_path.open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["pair_id", "reviewer_id", "preference", "notes"])
            writer.writeheader()
            for pair in packages:
                for _ in range(2):
                    writer.writerow({"pair_id": pair["pair_id"], "reviewer_id": "", "preference": "", "notes": ""})
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    valid = {}
    for row in rows:
        if not row.get("reviewer_id", "").strip() or row.get("preference") not in ("A", "B", "tie", "both_bad"):
            continue
        if row["pair_id"] not in key:
            raise ValueError("Unknown blind pair ID")
        identity = (row["pair_id"], row["reviewer_id"].strip())
        if identity in valid:
            raise ValueError("Duplicate reviewer for the same blind pair")
        valid[identity] = row
    complete = bool(packages) and all(sum(pair == p["pair_id"] for pair, reviewer in valid) >= 2 for p in packages)
    return {"status": "annotated" if complete else "awaiting_human_annotations",
            "pairs": len(packages), "valid_annotations": len(valid), "tasks_sha256": sha256(task_path),
            "annotations_sha256": sha256(csv_path), "not_population_level_claim": True}


def run_evaluations(models_override=None):
    models = model_inputs() if models_override is None else models_override
    EVAL.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(EVAL):
        path = EVAL / "state.json"
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"models": {}, "started_at": now()}
        state["evaluation_scope"] = {"benchmarks": list(TASKS), "human_review_required": False}
        atomic_json(path, state)
        for name, model in models.items():
            row = state["models"].setdefault(name, {})
            for key, argv, directory, result in jobs(name, model):
                try:
                    if row.get(key, {}).get("status") == "completed":
                        verify_record(row[key]["result"])
                        verify_evaluation_result(key, result, model)
                        if row[key].get("inputs") != model:
                            raise ValueError("Evaluation checkpoint/adapter identity changed")
                        continue
                    row[key] = {"status": "running", "inputs": model, "argv": argv}
                    atomic_json(path, state)
                    if not (key.startswith("tool_") and result.exists()):
                        code, log = execute(argv, directory)
                        if code:
                            raise RuntimeError(f"Evaluation exit {code}: {log}")
                    verify_evaluation_result(key, result, model)
                    row[key] = {"status": "completed", "inputs": model,
                                "result": {"path": str(result), "sha256": sha256(result)}}
                except Exception as error:
                    row[key].update(status="failed", error=str(error), traceback=traceback.format_exc())
                atomic_json(path, state)
        required = ({"full_sft", *(b["name"] for b in build_plan()["branches"])}
                    if models_override is None else set(models_override))
        machine_complete = set(models) == required and all(
            all(state["models"][name].get(key, {}).get("status") == "completed" for key, *_ in jobs(name, models[name]))
            for name in models)
        human = {"status": "cancelled_by_user", "required": False, "scores": None}
        state.update(status="machine_evaluations_complete" if machine_complete else "incomplete",
                     machine_evaluations_complete=machine_complete, human_review=human, updated_at=now())
        atomic_json(path, state)
        # This is evidence of workflow coverage, not proof every trained branch improved quality.
        acceptance = {"status": "accepted" if machine_complete else "pending",
            "required_training_branches": sorted(required), "machine_evaluations_complete": machine_complete,
            "human_review": human, "evaluation_state": str(path), "evaluation_state_sha256": sha256(path),
            "base": {"path": models["full_sft"]["checkpoint"], "sha256": models["full_sft"]["sha256"]},
            "claims": "Branch comparisons retain failures/degeneration; no guaranteed improvement, no invented human scores"}
        atomic_json(RUN / "text_acceptance.json", acceptance)
        return 0 if acceptance["status"] == "accepted" else 2


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(run_evaluations())
