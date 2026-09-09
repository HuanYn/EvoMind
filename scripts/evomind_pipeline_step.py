"""Fail-closed implementation of evomind's serial continuation stages.

Only this run's owned output directories can be resumed. No other jobs are
terminated, no model is downloaded implicitly, and an error is never hidden.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evomind_run import atomic_json, exclusive_run_lock, file_record, sha256

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "vision"
SOURCE_SPLIT = VISION / "dataset/evomind_split"
SPLIT = VISION / "dataset/evomind_controlled"
ENCODER = VISION / "model/siglip2-base-p32-256-ve"
BASE = VISION / "out/llm_768.pth"
TRAIN = VISION / "artifacts/training"
EVAL = VISION / "artifacts/evaluation"
OWNER = "evomind_vision_pipeline_20260908"


def run(argv, cwd=VISION):
    command = [sys.executable, "-u", *map(str, argv)]
    print(json.dumps({"event": "command", "cwd": str(cwd), "argv": command}, ensure_ascii=False), flush=True)
    from evomind_run import managed_child
    with managed_child(command, cwd=cwd) as child:
        code = child.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def create_or_compare(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing run contract changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)


def setup():
    provenance = ROOT / "artifacts/provenance/vision_assets.json"
    if not provenance.is_file():
        raise FileNotFoundError("Vision asset download/verification is not complete; inspect artifacts/setup/vision_http.*.log")
    assets = json.loads(provenance.read_text(encoding="utf-8"))
    if assets["status"] != "verified":
        raise ValueError("Vision assets must be verified before continuing")
    for item in assets["assets"]:
        path = Path(item["local_path"])
        if path.is_dir():
            for name, expected in item["files"].items():
                if sha256(path / name) != expected:
                    raise ValueError(f"Encoder asset changed: {path / name}")
        elif sha256(path) != item["sha256"]:
            raise ValueError(f"Dataset hash changed: {path}")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if sha256(ROOT / "model" / name) != sha256(VISION / "model" / name):
            raise ValueError(f"Text and vision tokenizers disagree: {name}")
    if (ROOT / "configs/product_pipeline.json").is_file():
        from evomind_product import selected_checkpoint
        source = selected_checkpoint()
    else:
        source = ROOT / "out/full_sft_768.pth"
    expected = sha256(source)
    BASE.parent.mkdir(parents=True, exist_ok=True)
    if BASE.exists():
        if sha256(BASE) != expected:
            raise FileExistsError(f"Refusing to overwrite a different vision base: {BASE}")
    else:
        temporary = BASE.with_suffix(".pth.tmp")
        if temporary.exists():
            raise FileExistsError(f"Previous incomplete transfer needs inspection: {temporary}")
        shutil.copy2(source, temporary)
        if sha256(temporary) != expected:
            raise ValueError("Text base transfer verification failed")
        os.replace(temporary, BASE)
    hashes = {}
    for directory in (ROOT / "scripts", VISION / "evomind_v", VISION / "scripts", VISION / "trainer"):
        for path in directory.glob("*.py"):
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    record = {"owner": OWNER, "source": file_record(source), "destination": file_record(BASE),
              "vision_assets_sha256": sha256(provenance), "continuation_source_hashes": hashes}
    target = ROOT / "artifacts/provenance/vision_base_transfer.json"
    create_or_compare(target, record)
    print("Vision assets, tokenizer and copied text base verified.", flush=True)


def prepare():
    summary_path = SOURCE_SPLIT / "summary.json"
    if summary_path.is_file():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if (data["seed"], data["val_fraction"], data["test_fraction"], data["max_rows"]) != (42, .02, .02, None):
            raise ValueError("Existing split does not match the full-source image-disjoint contract")
        if sha256(SOURCE_SPLIT / "manifest.jsonl") != data["manifest_sha256"]:
            raise ValueError("Prepared vision manifest changed")
        train_parquet = Path(data["official_train_parquet"])
        if not train_parquet.is_file() or sha256(train_parquet) != data["official_train_parquet_sha256"]:
            raise ValueError("Official train-only parquet is missing or changed")
        print("Verified existing full vision split; no re-extraction.", flush=True)
    else:
        run(["scripts/prepare_evomind_v_data.py", "--parquet", "dataset/sft_i2t.parquet",
             "--output-dir", SOURCE_SPLIT, "--seed", "42", "--val-fraction", "0.02", "--test-fraction", "0.02",
             "--official-train-parquet", "dataset/evomind_official_train.parquet"])
    run(["scripts/evomind_shuffle_manifest.py", "--input", SOURCE_SPLIT / "manifest.jsonl",
         "--output-dir", SPLIT, "--seed", "42"], ROOT)


def common(variant, seed, output):
    result = ["scripts/train_evomind_v.py", "--variant", variant,
              "--manifest", SPLIT / "manifest.jsonl", "--tokenizer", VISION / "model",
              "--vision-model", ENCODER, "--init-weights", BASE, "--allow-text-init",
              "--output-dir", output, "--seed", str(seed), "--hidden-size", "768", "--num-hidden-layers", "8",
              "--max-seq-len", "768", "--epochs", "2", "--batch-size", "1", "--grad-accum", "4",
              "--learning-rate", "5e-6", "--max-train-records", "20000", "--max-val-samples", "64",
              "--max-test-samples", "500", "--cache-eval-samples", "64", "--cache-eval-max-new-tokens", "128",
              "--save-every", "250", "--eval-every", "250", "--num-workers", "0",
              "--device", "cuda", "--dtype", "bfloat16"]
    if variant == "C":
        result += ["--cache-dir", VISION / "artifacts/features/fp32", "--cache-dtype", "float32"]
    return result


def probe():
    """At most one lower-memory fallback, and only for an observed CUDA OOM."""
    target = ROOT / "artifacts/preflight/vision_official.json"
    if target.exists():
        value = json.loads(target.read_text(encoding="utf-8"))
        if value.get("selected"):
            print("Using recorded successful vision preflight.", flush=True)
            return
        raise RuntimeError("Previous vision preflight failed; inspect it before retrying")
    observations = []
    for batch, accum in ((4, 1), (1, 4)):
        output = ROOT / f"artifacts/preflight/vision_official_b{batch}.json"
        if output.exists():
            result = json.loads(output.read_text(encoding="utf-8"))
            exit_code = result["exit_code"]
        else:
            from evomind_run import managed_child
            with managed_child([sys.executable, "-u", "scripts/evomind_probe_vision.py",
                                "--batch", str(batch), "--accumulation-steps", str(accum),
                                "--output", str(output)], cwd=ROOT) as child:
                exit_code = child.wait()
            if not output.is_file():
                raise RuntimeError(f"Vision probe exited {exit_code} without an auditable report")
            result = json.loads(output.read_text(encoding="utf-8"))
        observations.append(result)
        if exit_code == 0 and result.get("status") == "success":
            atomic_json(target, {"selected": {"batch_size": batch, "accumulation_steps": accum},
                                 "observations": observations, "fallback_policy": "Only one retry after confirmed OOM; effective batch remains four"})
            return
        if exit_code != 3 or result.get("status") != "oom":
            atomic_json(target, {"selected": None, "observations": observations})
            raise RuntimeError("Vision preflight failed for a reason other than OOM")
        print("Observed CUDA OOM. Probe process exited; attempting the single predefined B1/acc4 adaptation.", flush=True)
    atomic_json(target, {"selected": None, "observations": observations})
    raise RuntimeError("Both bounded vision batch probes ran out of memory")


def parity():
    target = VISION / "artifacts/parity/real_cache.json"
    if target.exists():
        result = json.loads(target.read_text(encoding="utf-8"))
        if (result.get("passed") is True and result.get("init_weights_sha256") == sha256(BASE)
                and result.get("manifest_sha256") == sha256(SPLIT / "manifest.jsonl")):
            print("Existing successful real cache parity evidence matches this base and manifest.", flush=True)
            return
        raise RuntimeError("Previous parity evidence failed or changed: inspect it, do not overwrite or relax tolerances")
    run(["scripts/verify_evomind_v_cache.py", "--manifest", SPLIT / "manifest.jsonl",
         "--vision-model", ENCODER, "--tokenizer", VISION / "model", "--init-weights", BASE,
         "--allow-text-init", "--output", target, "--max-seq-len", "768", "--samples", "3", "--device", "cuda"])


def cache():
    output = VISION / "artifacts/cache_preparation"
    summary = output / "cache_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        if data["manifest_sha256"] != sha256(SPLIT / "manifest.jsonl"):
            raise ValueError("Cache summary belongs to another manifest")
        print("Cache already prepared; actual tensor metadata/content checks remain enforced by consumers.", flush=True)
        return
    run([*common("C", 42, output), "--build-cache-only", "--cache-budget-gb", "32"])


def train(variant, seed, smoke):
    output = TRAIN / f"{variant}_seed{seed}"
    contract = {"owner": OWNER, "variant": variant, "seed": seed,
                "manifest_sha256": sha256(SPLIT / "manifest.jsonl"), "base_sha256": sha256(BASE),
                "fixed_epochs": 2, "max_train_records": 20000, "batch_size": 1, "grad_accum": 4,
                "max_seq_len": 768, "smoke_does_not_change_lr_horizon": True}
    owner_path = TRAIN / f"{variant}_seed{seed}.owner.json"
    if not owner_path.exists() and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Unowned nonempty training directory: {output}")
    create_or_compare(owner_path, contract)
    argv = common(variant, seed, output)
    checkpoint = output / "last.pt"
    if checkpoint.exists():
        argv += ["--resume", checkpoint]
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Interrupted before the first checkpoint; preserve this directory and inspect before restarting")
    if smoke:
        argv += ["--max-steps", "2"]
    run(argv)


def official():
    probe = json.loads((ROOT / "artifacts/preflight/vision_official.json").read_text(encoding="utf-8"))
    selection = probe.get("selected")
    if not selection:
        raise RuntimeError("No safe official vision batch found; inspect preflight instead of launching blindly")
    batch, accum = selection["batch_size"], selection["accumulation_steps"]
    if batch * accum != 4:
        raise ValueError("Official vision effective batch contract is four")
    run_dir = VISION / "artifacts/official_reference"
    argv = ["train_sft_vlm.py", "--from_weight", "llm", "--epochs", "2",
            "--batch_size", str(batch), "--accumulation_steps", str(accum), "--learning_rate", "5e-6",
            "--hidden_size", "768", "--num_hidden_layers", "8", "--max_seq_len", "768",
            "--freeze_llm", "1", "--dtype", "bfloat16", "--num_workers", "0",
            "--log_interval", "100", "--save_interval", "1000",
            "--data_path", "../dataset/evomind_official_train.parquet"]
    contract = {"owner": OWNER, "base_sha256": sha256(BASE), "argv": argv,
                "train_parquet_sha256": sha256(VISION / "dataset/evomind_official_train.parquet"),
                "adaptation": "Image-group holdout plus Windows workers=0; microbatch changes only if preflight requires."}
    contract_path = run_dir / "contract.json"
    checkpoint = VISION / "checkpoints/sft_vlm_768_resume.pth"
    if not contract_path.exists() and (checkpoint.exists() or (VISION / "out/sft_vlm_768.pth").exists()):
        raise FileExistsError("Refusing to use or overwrite an unowned official vision checkpoint")
    create_or_compare(contract_path, contract)
    if checkpoint.exists():
        argv += ["--from_resume", "1"]
    run(argv, cwd=VISION / "trainer")
    atomic_json(run_dir / "completed.json", {"completed_at": time.time(), "epochs": 2,
                "seed": 42, "weight": file_record(VISION / "out/sft_vlm_768.pth"), "contract": contract})


def evaluate(variant, seed):
    name = "official_reference" if variant == "official" else f"{variant}_seed{seed}"
    output = EVAL / name
    checkpoint = VISION / "out/sft_vlm_768.pth" if variant == "official" else TRAIN / name / "last.pt"
    sys.path.insert(0, str(VISION))
    from evomind_v.official_eval import examples
    _, official_source = examples(VISION)
    contract = {"owner": OWNER, "official_examples": official_source, "checkpoint_sha256": sha256(checkpoint),
                "manifest_sha256": sha256(SPLIT / "manifest.jsonl"), "split": "official_examples",
                "samples": 6, "max_new_tokens": 512, "seed": seed, "dtype": "bfloat16"}
    owner = EVAL / f"{name}.owner.json"
    if not owner.exists() and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite unowned evaluation artifacts: {output}")
    create_or_compare(owner, contract)
    summary = output / "summary.json"
    if summary.is_file():
        result = json.loads(summary.read_text(encoding="utf-8"))
        if (result.get("checkpoint_sha256") == contract["checkpoint_sha256"]
                and result.get("manifest_sha256") == contract["manifest_sha256"]
                and result.get("samples") == 6 and result.get("split") == "official_examples"
                and result.get("official_examples") == official_source
                and (output / "predictions.jsonl").is_file()):
            print("Previously completed evaluation matches this checkpoint and test protocol.", flush=True)
            return
        raise RuntimeError("Evaluation summary is incompatible or incomplete; preserve it for inspection")
    if output.exists() and any(output.iterdir()):
        archive_parent = VISION / "artifacts/interrupted"
        archive_parent.mkdir(parents=True, exist_ok=True)
        archive = archive_parent / f"{name}_{time.time_ns()}"
        if output.is_symlink() or output.resolve().parent != EVAL.resolve() or not archive.resolve().is_relative_to(VISION.resolve()):
            raise ValueError("Interrupted evaluation archive escaped the owned project paths")
        output.rename(archive)
        print(f"Preserved incomplete evaluation at {archive}; starting a fresh, same-protocol evaluation.", flush=True)
    argv = ["scripts/eval_evomind_v.py", "--output-dir", output, "--manifest", SPLIT / "manifest.jsonl",
            "--tokenizer", VISION / "model", "--vision-model", ENCODER,
            "--official-examples", "--split", "test", "--max-samples", "6", "--max-new-tokens", "512",
            "--seed", str(seed), "--device", "cuda", "--dtype", "bfloat16"]
    if variant == "official":
        argv += ["--official-weights", VISION / "out/sft_vlm_768.pth", "--max-seq-len", "768"]
    else:
        argv += ["--checkpoint", TRAIN / name / "last.pt"]
    run(argv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "prepare", "probe", "parity", "cache", "train", "official", "evaluate", "finalize"))
    parser.add_argument("--variant", choices=("A", "B", "C", "official"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action in ("probe", "parity", "cache", "train", "official", "evaluate"):
        from evomind_continue import require_vision_enabled
        require_vision_enabled()
    if args.action in ("train", "evaluate") and not args.variant:
        parser.error("Training/evaluation requires an explicit variant")
    if args.action == "train" and args.variant == "official":
        parser.error("Use official action for the native reference")
    lease = ROOT / "artifacts/runs/vision_device_lease"
    lease.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(lease):
        execute(args)


def execute(args):
    if args.action == "train":
        train(args.variant, args.seed, args.smoke)
    elif args.action == "evaluate":
        evaluate(args.variant, args.seed)
    elif args.action == "finalize":
        run(["scripts/evomind_aggregate.py", "--expected-seeds", "42", "123", "2026", "--expected-test-samples", "6", "--no-human-review"], ROOT)
        run(["scripts/evomind_report.py"], ROOT)
        run(["scripts/evomind_sync_notes.py"], ROOT)
        aggregate = json.loads((ROOT / "artifacts/evaluation/vision_aggregate.json").read_text(encoding="utf-8"))
        if not aggregate["controlled_comparison_ready"]:
            raise RuntimeError("Training/eval subprocesses ended, but controlled comparison checks did not pass; inspect aggregate and report")
        atomic_json(ROOT / "artifacts/runs/vision_pipeline_machine_complete.json", {
            "completed_at": time.time(), "machine_pipeline": "complete",
            "human_evaluation": "cancelled_by_user", "project_complete": False,
            "completion_scope": "Image workflow only; video implementation/evaluation and tool extension remain pending",
            "note": "This marker does not claim human ratings or successful quality improvement."})
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
