"""Prepare pinned, verified post-training assets; CPU/network only, never train.

Old HappyLLM inputs are read-only reusable sources. Every downloaded/copied file
is checked against its pinned Hub LFS SHA256 or Git blob SHA1, then SHA256 logged.
No dataset filtering, duplicate removal or tiny-subset substitution is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

from evomind_run import atomic_json, exclusive_run_lock, sha256

ROOT = Path(__file__).resolve().parents[1]
DATA_REV = "312afb4f76391145c6902f765bb51691c09a12f5"
RM_REV = "25f3593492ab4625ce00fce8c5e67802d6e702ca"
TEACHER_REV = "edba70ec15e06bc4280fbb96ac3383d73a7eab91"
TOKENIZER_REV = "f92512d4cd6142fa9acc0d6022375049a8974bf6"
OLD = Path("E:/project/Learning/HappyLLM")
DATA_FILES = {
    "dpo.jsonl": "ee934a8a455ccc99d1334d63e1254dd1d64f497fd067cfcbb71e3043f5b46768",
    "rlaif.jsonl": "8c6634db971fa34b0217f7db4f7c30684f57d20bbb771eb717f1b5aeacb089ba",
    "agent_rl.jsonl": "cb96bcc8096aecc5eccaab858f75d5ace1dc22da2302c2230457e29744a761ab",
    "lora_medical.jsonl": "abf66d2bf14bf5704f6c9f1a166061f55d95e4dfd03c29a3c33d085ccaca593f",
}


def tree(repo, revision, kind):
    url = f"https://huggingface.co/api/{'datasets' if kind == 'dataset' else 'models'}/{repo}/tree/{revision}?recursive=false&expand=false"
    with urllib.request.urlopen(url, timeout=60) as response:
        return {r["path"]: r for r in json.load(response) if r["type"] == "file"}


def valid(path, metadata):
    if not path.is_file() or path.stat().st_size != metadata["size"]:
        return False
    if metadata.get("lfs"):
        return sha256(path) == metadata["lfs"]["oid"]
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest() == metadata["oid"]


def get_file(repo, revision, name, dest, kind, metadata, reuse=None):
    from huggingface_hub import hf_hub_download
    target = dest / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not valid(target, metadata):
        raise ValueError(f"Existing target differs from pinned source; preserved for inspection: {target}")
    reused = None
    if not target.exists() and reuse is not None and valid(reuse, metadata):
        temporary = target.with_name(target.name + ".copying")
        shutil.copyfile(reuse, temporary)
        if not valid(temporary, metadata):
            raise ValueError(f"Copied bytes failed verification: {temporary}")
        os.replace(temporary, target)
        reused = str(reuse)
    if not target.exists():
        hf_hub_download(repo, filename=name, revision=revision, repo_type=kind, local_dir=str(dest))
    if not valid(target, metadata):
        raise ValueError(f"Pinned source integrity failed: {target}")
    result = {"repo": repo, "revision": revision, "filename": name,
              "path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target),
              "hub_file_metadata": metadata, "reused_readonly_source": reused}
    print(f"VERIFIED {target} ({result['bytes']:,} bytes)", flush=True)
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    for key, value in {"HF_HOME": str(ROOT / ".cache/huggingface"),
                       "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_XET": "1"}.items():
        os.environ.setdefault(key, value)
    report_dir = ROOT / "artifacts/provenance"
    report_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = ROOT / "artifacts/runs/posttrain_assets"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(lock_dir):
        if shutil.disk_usage(ROOT).free < 10 * 1024 ** 3:
            raise RuntimeError("At least 10 GiB free disk required for post-training preparation")
        started = time.time()
        assets = []
        data_tree = tree("jingyaogong/minimind_dataset", DATA_REV, "dataset")
        for name, expected in DATA_FILES.items():
            if data_tree[name].get("lfs", {}).get("oid") != expected:
                raise ValueError(f"Pinned dataset contract mismatch: {name}")
            old_subdir = {"dpo.jsonl": "minimind_dpo", "rlaif.jsonl": "minimind_rlaif"}.get(name)
            reuse = OLD / "data/raw" / old_subdir / name if old_subdir else None
            item = get_file("jingyaogong/minimind_dataset", DATA_REV, name, ROOT / "dataset", "dataset", data_tree[name], reuse)
            count, keys = 0, set()
            with Path(item["path"]).open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        raise ValueError(f"Unexpected blank record: {name}:{count + 1}")
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"Expected JSON object: {name}:{count + 1}")
                    count += 1
                    keys.update(record)
            item.update(records=count, fields=sorted(keys), transformation="none: full official file")
            assets.append(item)
        rm_tree = tree("internlm/internlm2-1_8b-reward", RM_REV, "model")
        for name, metadata in sorted(rm_tree.items()):
            if name == ".gitattributes":
                continue
            assets.append(get_file("internlm/internlm2-1_8b-reward", RM_REV, name,
                ROOT / "models/internlm2-1_8b-reward", "model", metadata,
                OLD / "models/internlm2-1_8b-reward" / name))
        teacher_tree = tree("jingyaogong/minimind-3-pytorch", TEACHER_REV, "model")
        name = "full_sft_768_moe.pth"
        if teacher_tree[name].get("lfs", {}).get("oid") != "a050020ea6d1b9e824693d0db525b1a0a8b40f36a934ea8d89da161368f20cc1":
            raise ValueError("Teacher source contract mismatch")
        assets.append(get_file("jingyaogong/minimind-3-pytorch", TEACHER_REV, name,
            ROOT / "models/teacher_official", "model", teacher_tree[name]))
        tok_tree = tree("jingyaogong/minimind-3", TOKENIZER_REV, "model")
        for name in ("tokenizer.json", "tokenizer_config.json"):
            item = get_file("jingyaogong/minimind-3", TOKENIZER_REV, name,
                ROOT / "models/teacher_official/tokenizer", "model", tok_tree[name])
            item["matches_local_bytes"] = sha256(ROOT / "model" / name) == item["sha256"]
            assets.append(item)
        report = {"status": "files_verified", "elapsed_seconds": time.time() - started,
            "assets": assets, "teacher_origin": "Public upstream pretrained MoE; NOT trained by evomind",
            "teacher_runtime_and_token_mapping_verified": False,
            "reward_remote_code_review": "pending; file integrity alone is not a code safety review",
            "not_training_complete": True}
        atomic_json(report_dir / "posttrain_assets.json", report)
        print("POSTTRAIN_ASSETS_READY (runtime compatibility still requires preflight)", flush=True)


if __name__ == "__main__":
    main()
