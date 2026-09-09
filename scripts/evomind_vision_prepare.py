"""CPU-only continuation of the already-running vision extraction.

Waits for its atomic completion marker, audits the full manifest, then performs
bounded-memory deterministic reordering. Never trains, downloads, builds GPU
features, or restarts an interrupted extractor. No GPU libraries are imported.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evomind_run import atomic_json, exclusive_run_lock, managed_child, now, pid_alive, sha256

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "configs/text_alignment_scope.json"
SPLIT = ROOT / "vision/dataset/evomind_split"
CONTROLLED = ROOT / "vision/dataset/evomind_controlled"
RUN = ROOT / "artifacts/runs/vision_cpu_prepare_20260908"
AUDIT = ROOT / "artifacts/provenance/vision_manifest_audit.json"


def require_cpu_scope(path=SCOPE):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("vision_cpu_preparation_enabled") is not True:
        raise RuntimeError("Vision CPU preparation is not explicitly enabled")


def available_memory():
    if os.name != "nt":
        raise RuntimeError("This resource-gated local preparer currently targets Windows")
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (key, ctypes.c_ulonglong) for key in ("total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended")]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return status.available


def cpu_environment():
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def validate_summary(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format_version") != 1:
        raise ValueError("Unsupported prepared manifest format")
    if (value.get("status"), value.get("seed"), value.get("val_fraction"), value.get("test_fraction"), value.get("max_rows")) != ("complete", 42, .02, .02, None):
        raise ValueError("Prepared data does not match full-source, image-disjoint seed42 contract")
    if value.get("grouping") != "sha256_original_image_bytes":
        raise ValueError("Expected original-image grouping")
    if value["counts"]["scanned"] != sum(item["total_rows"] for item in value["sources"]):
        raise ValueError("Full source was not scanned")
    for item in (path.parent / "manifest.jsonl", Path(value["official_train_parquet"])):
        if not item.is_file():
            raise FileNotFoundError(item)
    return value


def validate_source_contract(value):
    provenance = json.loads((ROOT / "artifacts/provenance/vision_assets.json").read_text(encoding="utf-8"))
    assets = [item for item in provenance["assets"] if item.get("filename") == "sft_i2t.parquet"]
    if provenance.get("status") != "verified" or len(assets) != 1 or len(value["sources"]) != 1:
        raise ValueError("Expected one verified official vision source")
    asset, source = assets[0], value["sources"][0]
    if (Path(source["path"]).resolve() != Path(asset["local_path"]).resolve()
            or source["sha256"] != asset["sha256"] or source["size_bytes"] != asset["bytes"]
            or source["total_rows"] != 2904511):
        raise ValueError("Prepared source differs from pinned vision dataset")
    if Path(value["official_train_parquet"]).resolve() != ROOT / "vision/dataset/evomind_official_train.parquet":
        raise ValueError("Official train export path differs from the owned target")


def verify_official_export(value):
    """Called only after RAM gate: read Arrow footers and hash the final export."""
    import pyarrow.parquet as pq
    source = pq.ParquetFile(value["sources"][0]["path"])
    target = Path(value["official_train_parquet"])
    exported = pq.ParquetFile(target)
    if (exported.metadata.num_rows != value["official_train_rows"]
            or not exported.schema_arrow.equals(source.schema_arrow)):
        raise ValueError("Official train export footer rows/schema mismatch")
    digest = sha256(target)
    if digest != value["official_train_parquet_sha256"]:
        raise ValueError("Official train export content hash mismatch")
    return {"path": str(target), "rows": exported.metadata.num_rows,
            "row_groups": exported.metadata.num_row_groups, "sha256": digest,
            "scope": "Footer/schema and complete file hash; rows not semantically redecoded"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-extractor-pid", type=int, required=True)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    args = parser.parse_args()
    if args.watch_extractor_pid <= 0 or args.min_free_gib < 1:
        parser.error("Use a positive observed extractor PID and at least 1GiB RAM headroom")
    require_cpu_scope()
    RUN.mkdir(parents=True, exist_ok=True)
    state = {"kind": "cpu_preparation_only", "pid": os.getpid(), "started_at": now(),
             "observed_extractor_pid": args.watch_extractor_pid, "vision_gpu_started": False,
             "minimum_available_bytes": int(args.min_free_gib * 1024 ** 3), "steps": {}}

    def update(status, **fields):
        state.update(status=status, updated_at=now(), **fields)
        atomic_json(RUN / "state.json", state)
        print(json.dumps({"status": status, **fields}, ensure_ascii=False), flush=True)

    def wait_resources():
        while True:
            require_cpu_scope()
            free = available_memory()
            if free >= state["minimum_available_bytes"]:
                if shutil.disk_usage(ROOT).free < 20 * 1024 ** 3:
                    raise RuntimeError("Less than 20GiB disk headroom; preserving current artifacts")
                return
            update("waiting_for_ram", available_bytes=free)
            time.sleep(30)

    def child(name, argv):
        wait_resources()
        source = ROOT / argv[0]
        if not source.is_file():
            raise FileNotFoundError(source)
        state["steps"][name] = {"status": "running", "argv": argv, "script_sha256": sha256(source)}
        update(name)
        with (RUN / f"{name}.log").open("a", encoding="utf-8") as log:
            with managed_child([sys.executable, "-B", "-u", *argv], cwd=ROOT, env=cpu_environment(),
                               stdout=log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)) as process:
                state["steps"][name]["pid"] = process.pid
                update(name)
                code = process.wait()
        state["steps"][name].update(status="complete" if code == 0 else "failed", exit_code=code)
        if code:
            raise RuntimeError(f"{name} failed ({code}); inspect {RUN / (name + '.log')}")

    with exclusive_run_lock(RUN):
        if (RUN / "state.json").exists():
            raise FileExistsError("An earlier CPU continuation has state; inspect it instead of overwriting or auto-retrying")
        try:
            summary_path = SPLIT / "summary.json"
            while not summary_path.is_file():
                require_cpu_scope()
                if not pid_alive(args.watch_extractor_pid):
                    raise RuntimeError("Observed extractor exited before its completion marker; do not re-extract over partial data")
                update("waiting_for_existing_extractor", summary=str(summary_path))
                time.sleep(30)
            source_summary = validate_summary(summary_path)
            validate_source_contract(source_summary)
            state["source_summary_sha256"] = sha256(summary_path)
            state["source_manifest_sha256"] = source_summary["manifest_sha256"]
            child("manifest_audit", ["scripts/evomind_audit_vision_manifest.py", "--input-summary", str(summary_path),
                                     "--output", str(AUDIT), "--verify-image-samples", "32"])
            audit = json.loads(AUDIT.read_text(encoding="utf-8"))
            if audit.get("status") != "pass":
                raise RuntimeError("Manifest integrity audit did not pass")
            if audit.get("manifest_sha256") != state["source_manifest_sha256"]:
                raise RuntimeError("Audit belongs to a different prepared manifest")
            wait_resources()
            update("verify_official_export")
            state["official_export"] = verify_official_export(source_summary)
            child("deterministic_shuffle", ["scripts/evomind_shuffle_manifest.py", "--input", str(SPLIT / "manifest.jsonl"),
                                             "--output-dir", str(CONTROLLED), "--seed", "42",
                                             "--expected-source-sha256", state["source_manifest_sha256"]])
            result = CONTROLLED / "summary.json"
            if not result.is_file():
                raise RuntimeError("Reordering did not produce its completion summary")
            shuffled = json.loads(result.read_text(encoding="utf-8"))
            if (shuffled.get("status") != "complete" or shuffled.get("seed") != 42
                    or shuffled.get("input_manifest_sha256") != state["source_manifest_sha256"]
                    or shuffled.get("records") != source_summary["counts"]["accepted"]
                    or any(shuffled.get("split_counts", {}).get(split, 0) != source_summary["counts"].get(f"{split}_records", 0)
                           for split in ("train", "val", "test"))):
                raise RuntimeError("Shuffled summary changed source, seed or split counts")
            update("complete", manifest_audit=str(AUDIT), controlled_summary=str(result),
                   controlled_summary_sha256=sha256(result),
                   note="CPU data ready only; GPU parity, feature cache, vision training and evaluation remain gated")
        except BaseException as error:
            update("failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
