"""Two-update CISPO hardware preflight, never a final training checkpoint."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

import evomind_posttrain as training
from evomind_run import atomic_json, exclusive_run_lock, now, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/runs/single3090_cispo_preflight_gpu2_20260909"
GPUS = {2: "GPU-3312e52e-57b1-ea7e-e2cf-fb4d86da3392"}
BASE = ROOT / "artifacts/runs/text_official_mini_20260908/snapshots/full_sft/full_sft_768.pth"


def prepare():
    plan = json.loads((ROOT / "configs/product_pipeline.json").read_text(encoding="utf-8"))
    branch = next(b for b in plan["stages"] if b["name"] == "cispo")
    files = [BASE, ROOT / branch["data"], Path(__file__), ROOT / "configs/product_pipeline.json"]
    for folder in ("trainer", "model", "models/internlm2-1_8b-reward"):
        files.extend(p for p in (ROOT / folder).glob("*") if p.is_file())
    files.extend(ROOT / name for name in ("dataset/lm_dataset.py", "scripts/evomind_posttrain.py", "scripts/evomind_run.py"))
    RUN.mkdir(parents=True, exist_ok=True)
    atomic_json(RUN / "manifest.json", {"created_at": now(), "probe_only": True,
        "branch": branch, "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
        "reason": "Check single GPU full configured G/length memory and update integration, not final selected-base training",
        "adaptation": "Reward CPU FP32 -> GPU2 FP32; policy/reference GPU2 BF16; SentencePiece 0.2.1 project override; no data/G/length reduction; user requests only GPU2"})
    print(RUN / "manifest.json", flush=True)


def run():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(GPUS.values()):
        raise RuntimeError("Must bind exactly authorized free physical GPU 2 by UUID")
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    with exclusive_run_lock(RUN):
        for relative, expected in manifest["files"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Input/code hash mismatch: {relative}")
        for index, uuid in GPUS.items():
            result = subprocess.check_output(["nvidia-smi", "-i", str(index),
                "--query-gpu=uuid,memory.used,gpu_recovery_action", "--format=csv,noheader,nounits"], text=True)
            actual, memory, recovery = map(str.strip, result.strip().split(","))
            if actual != uuid or int(memory) >= 500 or recovery != "None":
                raise RuntimeError(f"GPU {index} unavailable: {result}")
        branch = copy.deepcopy(manifest["branch"])
        branch["options"].update(reward_model_path=str(ROOT / "models/internlm2-1_8b-reward"),
            reward_device="cuda:0", reward_dtype="float32", device="cuda:0")
        branch["adaptations"].append(manifest["adaptation"])
        atomic_json(RUN / "effective_recipe.json", branch)
        for i, candidate in enumerate(branch["batch_candidates"]):
            directory = RUN / f"probe_{i}"
            if directory.exists():
                raise FileExistsError("Retain prior probe; do not overwrite or silently restart")
            code, log = training.execute(training.command(branch, candidate, directory, BASE, probe=True), directory)
            if code:
                if "out of memory" in log.read_text(encoding="utf-8").lower():
                    continue
                raise RuntimeError(f"Preflight failed; inspect {log}")
            receipt = training.check_receipt(branch, directory, probe=True)
            atomic_json(RUN / "result.json", {"status": "passed", "probe_only": True,
                "candidate": candidate, "receipt": receipt, "completed_at": now(),
                "next": "Wait for native DPO evaluation and chosen parent before full CISPO"})
            return
        raise RuntimeError("All configured batch candidates OOM; no length/G reduction allowed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
