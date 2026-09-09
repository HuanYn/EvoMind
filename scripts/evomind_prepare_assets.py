"""Download pinned public assets without modifying the previous project."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
TEXT_REV = "312afb4f76391145c6902f765bb51691c09a12f5"
VISION_REV = "1e279a8b665cb10383451a6af6fd62b9f35bdd79"
ENCODER_REV = "9465d1dc89db6bc6227c5b6b0e0ca9b940325d62"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def fetch(repo, revision, filename, directory, expected, reuse=None):
    from huggingface_hub import hf_hub_download
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if not target.exists() and reuse and reuse.exists():
        if digest(reuse) != expected:
            raise ValueError(f"Existing source hash mismatch: {reuse}")
        print(f"Copy verified existing data: {reuse} -> {target}", flush=True)
        shutil.copy2(reuse, target)
    if not target.exists():
        print(f"Download {repo}@{revision}/{filename}", flush=True)
        hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision,
                        filename=filename, local_dir=str(directory))
    actual = digest(target)
    if actual != expected:
        raise ValueError(f"Hash mismatch; preserving file for inspection: {target}: {actual}")
    print(f"Verified {target.name}: {target.stat().st_size:,} bytes", flush=True)
    return {"repo": repo, "revision": revision, "filename": filename,
            "local_path": str(target), "bytes": target.stat().st_size, "sha256": actual}

def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["text", "vision"])
    args = p.parse_args()
    records = []
    started = time.time()
    if args.mode == "text":
        for name, sha, reuse in [
            ("pretrain_t2t_mini.jsonl", "6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c", None),
            ("sft_t2t_mini.jsonl", "abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518", Path("E:/project/Learning/HappyLLM/data/raw/minimind_sft/sft_t2t_mini.jsonl")),
        ]:
            item = fetch("jingyaogong/minimind_dataset", TEXT_REV, name, ROOT / "dataset", sha, reuse)
            counts = {"records": 0, "nonempty": 0, "invalid_json": 0}
            with Path(item["local_path"]).open(encoding="utf-8") as f:
                for line in f:
                    counts["records"] += 1
                    if line.strip():
                        counts["nonempty"] += 1
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            counts["invalid_json"] += 1
            item.update(counts)
            if counts["invalid_json"]:
                raise ValueError(f"Invalid JSON in {name}: {counts}")
            records.append(item)
            print(f"Source counts: {name}: {counts}", flush=True)
    else:
        records.append(fetch("jingyaogong/minimind-v_dataset", VISION_REV, "sft_i2t.parquet",
                             ROOT / "vision" / "dataset", "712f4026cd0e21b369feddca7334b1e465cb8182b5f298006f3f4f877f926643"))
        from huggingface_hub import snapshot_download
        dest = ROOT / "vision" / "model" / "siglip2-base-p32-256-ve"
        snapshot_download("jingyaogong/siglip2-base-p32-256-ve", revision=ENCODER_REV,
                          local_dir=str(dest), allow_patterns=["*.json", "*.safetensors", "README.md"])
        model_sha = digest(dest / "model.safetensors")
        if model_sha != "c1e9cc19ed6704b87353ee00b9ff5d6191886d741898339984364f789c62810d":
            raise ValueError("Vision encoder weight hash mismatch")
        records.append({"repo": "jingyaogong/siglip2-base-p32-256-ve", "revision": ENCODER_REV,
                        "local_path": str(dest), "sha256": model_sha,
                        "files": {f.name: digest(f) for f in dest.iterdir() if f.is_file()}})
    report = ROOT / "artifacts" / "provenance" / f"{args.mode}_assets.json"
    atomic_json(report, {"status": "verified", "elapsed_seconds": time.time()-started, "assets": records})
    print(f"ASSETS_READY {report}", flush=True)

if __name__ == "__main__":
    main()
