"""CPU-only aggregation of recorded EvoMind-V experiments; never fabricate results.

Run from any directory. Defaults read vision/artifacts/{training,evaluation} and
write artifacts/evaluation/vision_aggregate.json, plots, cases, and blind packages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import statistics
import tempfile
from urllib.parse import quote

VARIANTS = ("A", "B", "C")
CONTROL_ARGUMENTS = ("seed", "epochs", "batch_size", "grad_accum", "learning_rate", "weight_decay",
                     "grad_clip", "dtype", "max_seq_len", "hidden_size", "num_hidden_layers", "allow_text_init")
CONTROL_HASHES = ("manifest_sha256", "init_weights_sha256", "tokenizer_sha256", "vision_fingerprint",
                  "initial_projector_sha256", "train_sample_ids_sha256", "val_sample_ids_sha256")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sample_ids_sha(rows):
    return hashlib.sha256("\n".join(row["sample_id"] for row in rows).encode()).hexdigest()


def read_json(path, warnings):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"{path}: {exc}")
        return None


def read_jsonl(path, warnings):
    rows = []
    if not Path(path).exists():
        warnings.append(f"Missing {path}")
        return rows
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                warnings.append(f"{path}:{number}: incomplete or invalid JSONL row was not counted")
    return rows


def atomic_json(value, path):
    atomic_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), path)


def atomic_text(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                     prefix=".aggregate-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def numeric(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def mean_std(values, expected=3):
    values = [float(value) for value in values if numeric(value)]
    return {"n": len(values), "expected_n": expected, "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
            "dispersion": "sample standard deviation, ddof=1; not a confidence interval",
            "status": "complete" if len(values) == expected else "pending"}


def latest_steps(events, event):
    return sorted({row["global_step"]: row for row in events
                   if row.get("event") == event and isinstance(row.get("global_step"), int)}.values(),
                  key=lambda row: row["global_step"])


def discover_training(training_root, warnings):
    runs = {}
    if not training_root.exists():
        return runs
    for directory in sorted(path for path in training_root.iterdir() if path.is_dir()):
        match = re.fullmatch(r"([ABC])_seed(\d+)", directory.name)
        if not match:
            continue
        config = read_json(directory / "run_config.json", warnings)
        events = read_jsonl(directory / "metrics.jsonl", warnings)
        variant, seed = match.group(1), int(match.group(2))
        arguments = (config or {}).get("arguments", {})
        issues = []
        if config is None:
            issues.append("missing_or_incomplete_run_config")
        if arguments and (arguments.get("variant") != variant or arguments.get("seed") != seed):
            issues.append("directory_and_run_config_disagree")
        terminal = [event for event in events if event.get("event") in ("complete", "probe_stopped")]
        last_terminal = terminal[-1] if terminal else {}
        train_events = latest_steps(events, "train")
        val_events = latest_steps(events, "validation")
        last_step = train_events[-1]["global_step"] if train_events else 0
        planned = (config or {}).get("planned_optimizer_steps")
        completed = bool(last_terminal.get("event") == "complete" and planned is not None and last_step == planned)
        if not completed:
            issues.append("training_not_complete")
        # Terminal wall durations cover separate smoke/resume processes. They
        # include validation/checkpoint work, but exclude model/data initialization.
        runtime_parts = [event.get("wall_seconds_this_process") for event in terminal]
        runtime = sum(runtime_parts) if runtime_parts and all(numeric(value) for value in runtime_parts) else None
        if terminal and terminal[-1].get("global_step", 0) < last_step:
            runtime = None
            issues.append("active_process_wall_time_not_finalized")
        peaks = [event.get("peak_allocated_bytes") for event in train_events if numeric(event.get("peak_allocated_bytes"))]
        identifier = f"{variant}_seed{seed}"
        runs[identifier] = {"run_id": identifier, "variant": variant, "seed": seed,
                            "directory": str(directory.resolve()), "config": config, "issues": issues,
                            "status": "complete" if completed and not issues else "pending",
                            "global_step": last_step, "planned_optimizer_steps": planned,
                            "train_runtime_seconds": runtime, "peak_allocated_bytes": max(peaks) if peaks else None,
                            "totals": last_terminal.get("totals"), "train_events": train_events, "validation_events": val_events,
                            "config_sha256": file_sha(directory / "run_config.json") if config else None}
    return runs


def discover_evaluations(evaluation_root, warnings, expected_samples=500):
    runs = {}
    if not evaluation_root.exists():
        return runs
    for summary_path in sorted(evaluation_root.rglob("summary.json")):
        summary = read_json(summary_path, warnings)
        if not summary or "heldout_sample_ids_sha256" not in summary:
            continue
        rows = read_jsonl(summary_path.parent / "predictions.jsonl", warnings)
        variant = summary.get("variant")
        seed = summary.get("training_seed")
        if variant in VARIANTS:
            if not isinstance(seed, int):
                warnings.append(f"{summary_path}: missing training_seed; cannot identify controlled run")
                continue
            identifier = f"{variant}_seed{seed}"
        else:
            identifier = "official_reference" if variant == "official_reference" else summary_path.parent.name
        issues = []
        if summary.get("split") not in ("test", "official_examples"):
            issues.append("not_final_test_split")
        if len(rows) != summary.get("samples"):
            issues.append("prediction_count_differs_from_summary")
        if len(rows) != expected_samples:
            issues.append(f"expected_{expected_samples}_final_test_samples")
        ids = [row.get("sample_id") for row in rows]
        if len(set(ids)) != len(ids) or any(identifier is None for identifier in ids):
            issues.append("duplicate_or_missing_prediction_sample_ids")
        elif sample_ids_sha(rows) != summary.get("heldout_sample_ids_sha256"):
            issues.append("heldout_sample_hash_mismatch")
        if not rows:
            issues.append("missing_predictions")
        elif all("ended_with_eos" in row and numeric(row.get("repeated_4gram_fraction")) for row in rows):
            observed = {"eos_rate": sum(bool(row["ended_with_eos"]) for row in rows) / len(rows),
                        "mean_repeated_4gram_fraction": statistics.mean(row["repeated_4gram_fraction"] for row in rows)}
            if any(not numeric(summary.get(key)) or abs(summary[key] - value) > 1e-10 for key, value in observed.items()):
                issues.append("prediction_metrics_disagree_with_summary")
        else:
            issues.append("predictions_missing_stop_or_repetition_metrics")
        if identifier in runs:
            warnings.append(f"Multiple evaluation outputs for {identifier}; requiring explicit evaluation-root disambiguation")
            runs[identifier]["issues"].append("ambiguous_multiple_evaluations")
            runs[identifier]["status"] = "pending"
            continue
        runs[identifier] = {"run_id": identifier, "variant": variant, "seed": seed,
                            "directory": str(summary_path.parent.resolve()), "summary": summary,
                            "predictions": rows, "issues": issues, "status": "complete" if not issues else "pending",
                            "summary_sha256": file_sha(summary_path),
                            "predictions_sha256": file_sha(summary_path.parent / "predictions.jsonl")
                            if (summary_path.parent / "predictions.jsonl").exists() else None}
    return runs


def compare_controls(training, evaluations, seed, variants=VARIANTS):
    identifiers = [f"{variant}_seed{seed}" for variant in variants]
    missing = [identifier for identifier in identifiers if identifier not in training or identifier not in evaluations]
    if missing:
        return {"seed": seed, "status": "pending", "comparable": False, "missing_runs": missing, "checks": {}}
    checks = {}
    configs = [(training[identifier].get("config") or {}) for identifier in identifiers]
    summaries = [evaluations[identifier]["summary"] for identifier in identifiers]

    def agree(name, values):
        checks[name] = {"match": all(value is not None for value in values) and all(value == values[0] for value in values),
                        "values": dict(zip(identifiers, values))}

    for key in CONTROL_HASHES:
        agree(key, [config.get(key) for config in configs])
    for key in CONTROL_ARGUMENTS:
        agree("argument_" + key, [config.get("arguments", {}).get(key) for config in configs])
    for key in ("planned_optimizer_steps", "train_records", "val_records"):
        agree(key, [config.get(key) for config in configs])
    for key in ("heldout_sample_ids_sha256", "max_new_tokens", "decoding", "split", "global_step"):
        agree("evaluation_" + key, [summary.get(key) for summary in summaries])
    for key in ("inference_dtype", "max_seq_len", "prompt_format"):
        agree("evaluation_" + key, [summary.get(key) for summary in summaries])
    for key in ("manifest_sha256", "vision_fingerprint"):
        checks["evaluation_matches_training_" + key] = {
            "match": all(summary.get(key) is not None and summary.get(key) == config.get(key)
                         for summary, config in zip(summaries, configs))}
    agree("actual_optimizer_steps", [training[identifier]["global_step"] for identifier in identifiers])
    for key in ("samples", "optimizer_steps"):
        agree("actual_train_" + key, [(training[identifier].get("totals") or {}).get(key) for identifier in identifiers])
    finished = all(training[identifier]["status"] == "complete" and evaluations[identifier]["status"] == "complete"
                   for identifier in identifiers)
    matches = all(check["match"] for check in checks.values())
    return {"seed": seed, "status": "complete" if finished and matches else "pending" if not finished else "incompatible",
            "comparable": finished and matches, "checks": checks,
            "note": "A/B compare visual input; B/C compare caching. Equal selection, initialization, budgets and decoding are required."}


def variant_statistics(training, evaluations, seeds, expected):
    result = {}
    eval_metrics = ("eos_rate", "mean_repeated_4gram_fraction", "mean_generation_seconds", "wall_seconds",
                    "peak_allocated_bytes", "closed_answer_exact_match", "ocr_character_error_rate")
    for variant in VARIANTS:
        completed_training = [training[f"{variant}_seed{seed}"] for seed in seeds
                              if f"{variant}_seed{seed}" in training and training[f"{variant}_seed{seed}"]["status"] == "complete"]
        completed_eval = [evaluations[f"{variant}_seed{seed}"] for seed in seeds
                          if f"{variant}_seed{seed}" in evaluations and evaluations[f"{variant}_seed{seed}"]["status"] == "complete"]
        result[variant] = {"training_seeds": [run["seed"] for run in completed_training],
                           "evaluation_seeds": [run["seed"] for run in completed_eval],
                           "train_runtime_seconds": mean_std([run["train_runtime_seconds"] for run in completed_training], expected),
                           "train_peak_allocated_bytes": mean_std([run["peak_allocated_bytes"] for run in completed_training], expected),
                           "evaluation": {key: mean_std([run["summary"].get(key) for run in completed_eval], expected)
                                          for key in eval_metrics}}
    return result


def compare_bc_predictions(training, evaluations, seed=42):
    controls = compare_controls(training, evaluations, seed, variants=("B", "C"))
    result = {"seed": seed, "status": controls["status"], "controls": controls,
              "interpretation": "Observed agreement/differences for one seed and a finite test set; not proof of general or bitwise equivalence."}
    keys = (f"B_seed{seed}", f"C_seed{seed}")
    if any(key not in evaluations for key in keys):
        return result
    first, second = ({row["sample_id"]: row for row in evaluations[key]["predictions"]} for key in keys)
    result["only_in_B"] = sorted(first.keys() - second.keys())
    result["only_in_C"] = sorted(second.keys() - first.keys())
    if not controls["comparable"] or result["only_in_B"] or result["only_in_C"]:
        result["comparison_performed"] = False
        return result
    rows = []
    for identifier in first:
        left, right = first[identifier], second[identifier]
        context_match = left.get("conversation_context") == right.get("conversation_context") and left.get("image_hash") == right.get("image_hash")
        rows.append({"sample_id": identifier, "context_matches": context_match,
                     "text_equal": left.get("prediction") == right.get("prediction"),
                     "token_ids_equal": left.get("generated_token_ids") == right.get("generated_token_ids"),
                     "eos_equal": left.get("ended_with_eos") == right.get("ended_with_eos"),
                     "C_minus_B_repeated_4gram_fraction": right["repeated_4gram_fraction"] - left["repeated_4gram_fraction"]})
    if any(not row["context_matches"] for row in rows):
        result.update(status="incompatible", comparison_performed=False, context_mismatch_count=sum(not row["context_matches"] for row in rows))
        return result
    result.update(comparison_performed=True, samples=len(rows), per_sample=rows,
                  text_differences=sum(not row["text_equal"] for row in rows),
                  token_sequence_differences=sum(not row["token_ids_equal"] for row in rows),
                  eos_differences=sum(not row["eos_equal"] for row in rows),
                  mean_C_minus_B_repetition=statistics.mean(row["C_minus_B_repeated_4gram_fraction"] for row in rows))
    return result


def cache_amortization(artifact_root, training, controls, warnings):
    records = []
    for path in sorted(artifact_root.rglob("cache_summary.json")) if artifact_root.exists() else []:
        value = read_json(path, warnings)
        if value and value.get("event") == "cache_complete":
            records.append({"path": str(path.resolve()), **value})
    by_fingerprint = {}
    for row in records:
        metadata = row.get("metadata", {})
        key = (metadata.get("fingerprint"), metadata.get("dtype"), metadata.get("mode"))
        by_fingerprint.setdefault(key, []).append(row)
    result = {"status": "pending", "preparation_records": records, "comparisons": [],
              "runtime_definition": "Recorded per-process training-loop wall time, including validation/checkpoint work; model/data initialization excluded. Not pure kernel throughput."}
    eligible = [entry["seed"] for entry in controls if entry.get("comparable")]
    if not eligible:
        return result
    bc = [(training[f"B_seed{seed}"], training[f"C_seed{seed}"]) for seed in eligible]
    c_config = bc[0][1]["config"]
    cache_key = (c_config["vision_fingerprint"], c_config["arguments"].get("cache_dtype"), "multi")
    matching = by_fingerprint.get(cache_key, [])
    complete_history = bool(matching and any(row.get("existing") == 0 and row.get("created", 0) > 0 for row in matching))
    preparation = sum(row["elapsed_seconds"] for row in matching if numeric(row.get("elapsed_seconds"))) if complete_history else None
    result.update(cache_cost_history_starts_empty=complete_history, recorded_preparation_seconds=preparation,
                  matching_preparation_paths=[row["path"] for row in matching])
    usable = [(b, c) for b, c in bc if numeric(b["train_runtime_seconds"]) and numeric(c["train_runtime_seconds"])]
    if preparation is None or not usable:
        return result
    # Report actual sums for one recorded run and for three distinct seed runs.
    for count in (1, 3):
        chosen = usable[:count]
        if len(chosen) != count:
            result["comparisons"].append({"reuses": count, "status": "pending"})
            continue
        online = sum(b["train_runtime_seconds"] for b, c in chosen)
        cached_train = sum(c["train_runtime_seconds"] for b, c in chosen)
        result["comparisons"].append({"reuses": count, "status": "complete", "seeds": [b["seed"] for b, c in chosen],
                                       "B_online_total_seconds": online, "C_training_only_seconds": cached_train,
                                       "cache_preparation_seconds_once": preparation,
                                       "C_end_to_end_total_seconds": preparation + cached_train,
                                       "C_amortized_seconds_per_reuse": (preparation + cached_train) / count,
                                       "observed_time_saved_seconds": online - preparation - cached_train})
    result["status"] = "complete" if all(row["status"] == "complete" for row in result["comparisons"]) else "pending"
    return result


def plot_curves(training, output_dir, warnings):
    if not any(run["train_events"] for run in training.values()):
        return {"status": "pending", "files": []}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        warnings.append(f"Plots pending: {exc}")
        return {"status": "pending", "files": []}
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = (("train_answer_nll.png", "train_events", "answer_nll", "Training answer NLL (nats / target token)"),
                   ("validation_answer_nll.png", "validation_events", "answer_nll", "Validation answer NLL (nats / target token)"),
                   ("learning_rate.png", "train_events", "learning_rate", "Learning rate"))
    files = []
    for filename, collection, key, label in definitions:
        figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        count = 0
        for name, run in sorted(training.items()):
            rows = [row for row in run[collection] if numeric(row.get(key))]
            if rows:
                axis.plot([row["global_step"] for row in rows], [row[key] for row in rows], linewidth=1, label=name, alpha=0.8)
                count += 1
        if count:
            axis.set(xlabel="Optimizer update", ylabel=label)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, ncol=3)
            path = output_dir / filename
            figure.savefig(path, dpi=160)
            files.append(str(path.resolve()))
        plt.close(figure)
    return {"status": "complete" if len(files) == 3 else "pending", "files": files,
            "note": "Recorded unsmoothed points; duplicate step logs use their last recorded occurrence. Partial runs are plotted as observed."}


def input_signature(row):
    context = row.get("conversation_context")
    if not isinstance(context, list):
        return None
    return digest({"image_hash": row.get("image_hash"), "conversation_context": context})


def common_examples(evaluations):
    usable = {name: run for name, run in evaluations.items() if run["status"] == "complete"}
    if not usable:
        return usable, []
    indexed = {}
    for name, run in usable.items():
        records = {}
        for row in run["predictions"]:
            signature = input_signature(row)
            if signature is not None:
                records.setdefault(signature, row)
        indexed[name] = records
    shared = set.intersection(*(set(rows) for rows in indexed.values()))
    return usable, [(signature, {name: rows[signature] for name, rows in indexed.items()}) for signature in sorted(shared)]


def export_blind(evaluations, output_dir, *, image_count=30, selection_seed=42, readiness=False):
    usable, common = common_examples(evaluations)
    if not readiness or not common:
        return {"status": "pending", "reason": "Need completed, comparable A/B/C runs and shared final-test contexts", "human_raters": 0}
    ranked = sorted(common, key=lambda item: digest([selection_seed, item[0]]))
    selected, seen = [], set()
    for signature, responses in ranked:
        first = next(iter(responses.values()))
        if first["image_hash"] not in seen:
            selected.append((signature, responses))
            seen.add(first["image_hash"])
        if len(selected) == image_count:
            break
    if len(selected) < image_count:
        return {"status": "pending", "reason": f"Only {len(selected)} common unique test images; requested {image_count}", "human_raters": 0}
    version = digest({"selection": [signature for signature, _ in selected],
                      "runs": {name: run["predictions_sha256"] for name, run in sorted(usable.items())}})[:16]
    package_dir = output_dir / "blind_packages" / version
    key_path = output_dir / "blind_keys" / f"{version}.json"
    if package_dir.exists():
        return {"status": "ready_for_manual_annotation", "package_dir": str(package_dir.resolve()),
                "key_path": str(key_path.resolve()), "images": image_count, "responses": image_count * len(usable),
                "human_ratings_imported": False, "note": "Existing package preserved, including any manual edits; no human scores were imported."}
    for _, responses in selected:
        first = next(iter(responses.values()))
        path = Path(first["image_path"])
        if not path.is_file() or file_sha(path) != first["image_hash"]:
            return {"status": "pending", "reason": f"Missing or changed original image for blind package: {path}", "human_raters": 0}
    package_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".blind-", dir=package_dir.parent))
    (staging / "images").mkdir()
    public_rows, private_rows = [], []
    for index, (_, responses) in enumerate(selected, start=1):
        item_id = f"item_{index:03d}"
        first = next(iter(responses.values()))
        source = Path(first["image_path"])
        image_name = f"images/{item_id}{source.suffix}"
        shutil.copy2(source, staging / image_name)
        for run_id, row in responses.items():
            response_id = secrets.token_hex(12)
            public_rows.append({"response_id": response_id, "item_id": item_id, "image_file": image_name,
                                "question": row["question"], "conversation_context": json.dumps(row["conversation_context"], ensure_ascii=False),
                                "prediction": row["prediction"]})
            private_rows.append({"response_id": response_id, "item_id": item_id, "run_id": run_id,
                                 "sample_id": row["sample_id"], "image_hash": row["image_hash"]})
    random.SystemRandom().shuffle(public_rows)
    columns = ["response_id", "item_id", "image_file", "question", "conversation_context", "prediction",
               "correctness_1_to_5", "grounding_1_to_5", "hallucination_yes_no", "notes"]
    with (staging / "annotation.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(public_rows)
    (staging / "README.txt").write_text(
        "Open annotation.csv with the images folder beside it. Use the image and full visible conversation context.\n"
        "Scores are intentionally blank. A person must supply any ratings. Do not infer model identity.\n"
        "1 = poor and 5 = strong for correctness/grounding. Mark hallucination yes/no and explain uncertain cases.\n",
        encoding="utf-8")
    atomic_json({"package_version": version, "selection_seed": selection_seed, "mappings": private_rows,
                 "warning": "Keep this key and aggregate report separate from raters."}, key_path)
    os.replace(staging, package_dir)
    return {"status": "ready_for_manual_annotation", "package_dir": str(package_dir.resolve()),
            "key_path": str(key_path.resolve()), "images": image_count, "responses": len(public_rows),
            "human_raters": 0, "human_ratings_imported": False,
            "note": "Anonymous response IDs and one shared prompt per unique image; rows shuffled across all runs. Share package only, not key/report."}


def export_cases(evaluations, output_path):
    usable, common = common_examples(evaluations)
    if not usable:
        return {"status": "pending", "path": None}
    sections = ["# Recorded test examples", "", "These are actual generated outputs, not correctness judgments. Selection is deterministic and is not a best-case ranking.", ""]

    def render(title, responses):
        first = next(iter(responses.values()))
        image_path = Path(first["image_path"])
        relative = quote(os.path.relpath(image_path, output_path.parent).replace("\\", "/"), safe="/:._-+")
        sections.extend([f"## {title}", "", f"![Original test image]({relative})", "",
                         "Conversation context:", "", "```json", json.dumps(first.get("conversation_context", []), ensure_ascii=False, indent=2), "```", ""])
        for name, row in sorted(responses.items()):
            sections.extend([f"### {name}", "", f"Sample: `{row['sample_id']}`", "", "```text", row["prediction"].replace("```", "'''"), "```", ""])

    if common:
        for index, (_, responses) in enumerate(common[:3], start=1):
            render(f"Shared input {index}", responses)
        policy = "first_three_shared_image_and_full_context_signatures"
    else:
        sections.extend(["No shared complete input contexts were found; each run shows its first three recorded test examples.", ""])
        for name, run in sorted(usable.items()):
            for index, row in enumerate(run["predictions"][:3], start=1):
                render(f"{name} example {index}", {name: row})
        policy = "first_three_real_records_per_run_no_shared_input"
    atomic_text("\n".join(sections), output_path)
    return {"status": "complete", "path": str(output_path.resolve()), "selection_policy": policy}


def aggregate(project_root, *, output_dir=None, expected_seeds=None, expected_seed_count=3,
              image_count=30, selection_seed=42, comparison_seed=42, expected_test_samples=500, human_review=True):
    project_root = Path(project_root).resolve()
    artifact_root = project_root / "vision" / "artifacts"
    output_dir = Path(output_dir).resolve() if output_dir else project_root / "artifacts" / "evaluation"
    warnings = []
    training = discover_training(artifact_root / "training", warnings)
    evaluations = discover_evaluations(artifact_root / "evaluation", warnings, expected_test_samples)
    seeds = sorted(set(expected_seeds) if expected_seeds is not None else
                   {run["seed"] for run in training.values()} | {run["seed"] for run in evaluations.values() if run["variant"] in VARIANTS})
    expected = len(seeds) if expected_seeds is not None else expected_seed_count
    controls = [compare_controls(training, evaluations, seed) for seed in seeds]
    ready = len(seeds) == expected and all(entry["comparable"] for entry in controls)
    reference_ready = evaluations.get("official_reference", {}).get("status") == "complete"
    result = {"schema_version": 1, "status": "complete" if ready and reference_ready else "pending", "observed_or_expected_seeds": seeds,
              "expected_seed_count": expected, "warnings": warnings, "controlled_comparison_ready": ready,
              "expected_test_samples": expected_test_samples, "official_reference_ready": reference_ready,
              "statistical_policy": f"Means and sample standard deviations across independent training seeds; expected n={expected}. These are not confidence intervals. Descriptive statistics remain provisional until control checks pass.",
              "controls": controls, "variant_statistics": variant_statistics(training, evaluations, seeds, expected),
              "B_C_single_seed": compare_bc_predictions(training, evaluations, comparison_seed),
              "cache_amortization": cache_amortization(artifact_root, training, controls, warnings),
              "plots": plot_curves(training, output_dir / "plots", warnings),
              "blind_evaluation": export_blind(evaluations, output_dir, image_count=image_count,
                                                selection_seed=selection_seed, readiness=ready and reference_ready) if human_review else {"status": "cancelled_by_user", "human_raters": 0},
              "cases": export_cases(evaluations, output_dir / "cases.md"),
              "training_runs": {name: {key: value for key, value in run.items() if key not in ("train_events", "validation_events", "config")}
                                for name, run in training.items()},
              "evaluation_runs": {name: {key: value for key, value in run.items() if key != "predictions"}
                                  for name, run in evaluations.items()},
              "official_reference_policy": "Official reference may have a different training budget and native prompt format; do not present it as a controlled A/B/C gain.",
              "data_limitations": "Original image SHA256 groups byte-identical files, not all visually identical re-encodings. Open-caption exact accuracy and simulated human scores are not reported."}
    destination = output_dir / "vision_aggregate.json"
    atomic_json(result, destination)
    return result, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-seeds", "--seeds", dest="expected_seeds", type=int, nargs="+")
    parser.add_argument("--expected-seed-count", type=int, default=3)
    parser.add_argument("--expected-test-samples", type=int, default=500)
    parser.add_argument("--blind-images", type=int, default=30)
    parser.add_argument("--no-human-review", action="store_true")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--comparison-seed", type=int, default=42)
    args = parser.parse_args()
    if args.expected_seed_count <= 0 or args.expected_test_samples <= 0 or args.blind_images <= 0 or args.expected_seeds is not None and len(set(args.expected_seeds)) != len(args.expected_seeds):
        parser.error("Counts must be positive and expected seed IDs unique")
    result, destination = aggregate(args.project_root, output_dir=args.output_dir, expected_seeds=args.expected_seeds,
                                    expected_seed_count=args.expected_seed_count, image_count=args.blind_images,
                                    selection_seed=args.selection_seed, comparison_seed=args.comparison_seed,
                                    expected_test_samples=args.expected_test_samples, human_review=not args.no_human_review)
    print(json.dumps({"aggregate": str(destination), "status": result["status"],
                      "controlled_comparison_ready": result["controlled_comparison_ready"],
                      "observed_or_expected_seeds": result["observed_or_expected_seeds"], "warnings": result["warnings"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
