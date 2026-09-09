"""Stage a verified SFT-to-DPO handoff; run only DPO on an authorized server GPU.

Local evaluations remain authoritative under their original harness contract.
The remote output MUST return for candidate evaluation before CISPO/vision.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

import evomind_posttrain as training
from evomind_run import atomic_json, exclusive_run_lock, now, sha256

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = "artifacts/deployment/gpu3_20260909_retry1"
REMOTE = "/data/yinhuan/evomind"
GPU_UUID = "GPU-d4436045-2e7c-690f-b03f-92b2fe9abb0e"


def prepare():
    import evomind_product as product
    plan = product.settings()
    product.configure_paths(plan)
    base = training.base_checkpoint()
    base_record = {"path": str(base), "sha256": sha256(base)}
    scores, evidence = product.evaluate({"full_sft": product.model(base_record)})
    if not product.baseline_ready(scores["full_sft"], plan["selection"]):
        raise RuntimeError("SFT screening gate did not pass")
    branch = next(b for b in plan["stages"] if b["name"] == "dpo")
    training.validate_assets(branch)
    files = set()
    for folder in ("model", "trainer"):
        files.update(p for p in (ROOT / folder).glob("*") if p.suffix in (".py", ".json"))
    files.update((ROOT / "dataset").glob("*.py"))
    files.update(ROOT / p for p in (
        "scripts/evomind_remote_dpo.py", "scripts/evomind_posttrain.py", "scripts/evomind_run.py",
        "configs/product_pipeline.json", "LICENSE", "dataset/dpo.jsonl"))
    files.add(base)
    for item in evidence:
        path = training.verify_record(item)
        files.add(path)
        for sibling in ("results.json", "records.jsonl", "contract.json"):
            candidate = path.parent / sibling
            if candidate.is_file():
                files.add(candidate)
    provenance = json.loads((ROOT / "artifacts/provenance/posttrain_assets.json").read_text(encoding="utf-8"))
    asset = next(a for a in provenance["assets"] if a["filename"] == "dpo.jsonl")
    asset = {**asset, "original_local_path": asset["path"], "path": REMOTE + "/dataset/dpo.jsonl"}
    payload = {"status": "files_verified", "assets": [asset], "deployment_only": True}
    manifest = {"created_at": now(), "host": "10.10.16.18", "remote_root": REMOTE,
        "gpu_uuid": GPU_UUID, "physical_gpu": 3, "branch": branch,
        "base": {"relative_path": base.relative_to(ROOT).as_posix(), "sha256": sha256(base)},
        "gate": {"status": "verified_local_native_evaluations", "scores": scores,
                 "selection_policy": plan["selection"], "evidence": evidence},
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)},
        "adaptation": "Linux single RTX3090; same data/epochs/global batch/loss; Torch version recorded by native runtime, not bitwise equivalent",
        "next_gate": "Return DPO weights for native evaluation; no automatic CISPO or vision promotion"}
    dest = ROOT / DEPLOY
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "deployment.json"
    if target.exists():
        raise FileExistsError("Preserve existing deployment; inspect rather than overwrite")
    atomic_json(target, manifest)
    archive = dest / "payload.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        for p in sorted(files):
            out.add(p, arcname=p.relative_to(ROOT).as_posix(), recursive=False)
        out.add(target, arcname=f"{DEPLOY}/deployment.json")
        content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
        info = tarfile.TarInfo("artifacts/provenance/posttrain_assets.json")
        info.size = len(content)
        out.addfile(info, io.BytesIO(content))
    atomic_json(dest / "transfer.json", {"archive": str(archive), "sha256": sha256(archive),
                "bytes": archive.stat().st_size, "deployment_sha256": sha256(target)})
    print(json.dumps({"archive": str(archive), "gate": "passed", "sha256": sha256(archive)}), flush=True)


def run():
    manifest_path = ROOT / DEPLOY / "deployment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(ROOT) != manifest["remote_root"] or os.environ.get("CUDA_VISIBLE_DEVICES") != manifest["gpu_uuid"]:
        raise RuntimeError("Remote root or authorized GPU binding differs")
    for relative, expected in manifest["files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise ValueError(f"Deployment file mismatch: {relative}")
    if manifest["gate"]["status"] != "verified_local_native_evaluations":
        raise RuntimeError("No verified baseline handoff")
    gpu = subprocess.check_output(["nvidia-smi", "-i", str(manifest["physical_gpu"]),
        "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    observed_uuid, memory = [v.strip() for v in gpu.split(",")]
    if observed_uuid != manifest["gpu_uuid"]:
        raise RuntimeError("Physical GPU mapping changed")
    if int(memory) >= 500:
        raise RuntimeError("GPU is now occupied; refusing to launch")
    training.RUN = ROOT / "artifacts/runs/text_product_remote3090_20260909"
    training.RUN.mkdir(parents=True, exist_ok=True)
    base = ROOT / manifest["base"]["relative_path"]
    with exclusive_run_lock(training.RUN):
        state_path = training.RUN / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "deployment_sha256": sha256(manifest_path), "base": {"path": str(base), "sha256": sha256(base)},
            "branches": {}, "created_at": now()}
        if state["deployment_sha256"] != sha256(manifest_path):
            raise RuntimeError("Deployment changed across resume")
        def persist():
            state["updated_at"] = now()
            atomic_json(state_path, state)
        try:
            state["status"] = "running_dpo"
            persist()
            training.run_branch(manifest["branch"], state, persist, base)
            state["status"] = "dpo_complete_awaiting_evaluation"
            persist()
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            persist()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
