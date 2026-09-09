"""Full upstream benchmark tasks through installed lm-evaluation-harness.

One task/group per resumable job; no sample limit or invented local MCQ template.
Use the strict native HF-compatible model (including an optional LoRA adapter).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from evomind_run import atomic_json, sha256
from evomind_load_text_model import ROOT, load_model
from evomind_harness_data import identity as dataset_identity, local_chinese_datasets

TASKS = ("ceval-valid", "cmmlu", "arc_easy", "piqa", "openbookqa", "hellaswag", "social_iqa")


def harness_identity():
    spec = importlib.util.find_spec("lm_eval")
    if spec is None:
        raise ImportError("lm-evaluation-harness must be installed in the project environment")
    root = Path(spec.origin).parent
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.suffix in (".py", ".yaml", ".yml")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return {"version": importlib.metadata.version("lm_eval"), "source_sha256": digest.hexdigest()}


def contract(checkpoint, lora, task):
    if task not in TASKS:
        raise ValueError("Not a configured upstream benchmark")
    return {"schema_version": 2, "dataset_loading": dataset_identity(task),
            "checkpoint_sha256": sha256(checkpoint), "lora_sha256": sha256(lora) if lora else None,
            "task": task, "harness": harness_identity(), "limit": None, "num_fewshot": "task_default",
            "apply_chat_template": True, "chat_template_args": {"open_thinking": False},
            "batch_size": 1, "dtype": "float16", "backend": "hf_preinitialized_causal",
            "seed": 42, "bootstrap_iters": 100000,
            "evaluator_sha256": sha256(__file__), "loader_sha256": sha256(ROOT / "scripts/evomind_load_text_model.py"),
            "model_sha256": sha256(ROOT / "model/model_minimind.py"),
            "tokenizer_sha256": {p.name: sha256(p) for p in sorted((ROOT / "model").glob("*.json"))}}


def verify_result(path, expected):
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("contract") != expected:
        raise ValueError("Harness evaluation is incomplete or its contract changed")
    raw_path = Path(path).parent / "results.json"
    if sha256(raw_path) != summary.get("results_sha256"):
        raise ValueError("Harness raw result changed")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    validate_coverage(raw)
    if summary.get("scores") != raw["results"] or summary.get("n_samples") != raw["n-samples"]:
        raise ValueError("Harness summary differs from raw scores/counts")
    return summary


def validate_coverage(result):
    if not result.get("results") or not result.get("samples") or not result.get("n-samples"):
        raise ValueError("Missing harness scores, per-document outputs or coverage")
    if result.get("config", {}).get("limit") is not None:
        raise ValueError("Limited harness runs cannot satisfy full evaluation")
    for name, count in result["n-samples"].items():
        original, effective = count.get("original"), count.get("effective")
        if not isinstance(original, int) or original < 1 or effective != original:
            raise ValueError("Harness split coverage is incomplete")
        rows = result["samples"].get(name, [])
        if len(rows) != effective or len({r["doc_id"] for r in rows}) != effective:
            raise ValueError("Missing/duplicate per-document harness outputs")


def serializable(value):
    # A preinitialized HFLM records torch.dtype in its config. It is metadata,
    # not a score; don't stringify arbitrary unknown values and conceal bugs.
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    raise TypeError(f"Unsupported harness result type: {type(value)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lora")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    expected = contract(args.checkpoint, args.lora, args.task)
    if (dest / "contract.json").exists():
        if json.loads((dest / "contract.json").read_text(encoding="utf-8")) != expected:
            raise ValueError("Refusing to mix harness protocols or checkpoints")
    else:
        if (dest / "results.json").exists() or (dest / "summary.json").exists():
            raise ValueError("Unowned harness output; choose a fresh directory")
        atomic_json(dest / "contract.json", expected)
    if (dest / "summary.json").exists():
        verify_result(dest / "summary.json", expected)
        print("Validated complete harness task; no model loaded", flush=True)
        return
    import datasets  # Windows DLL order
    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    model, tokenizer = load_model(args.checkpoint, args.lora)
    model = model.to(dtype=torch.float16)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", batch_size=1,
              chat_template_args={"open_thinking": False})
    with local_chinese_datasets(args.task):
        result = simple_evaluate(model=lm, tasks=[args.task], apply_chat_template=True,
                                 limit=None, log_samples=True, random_seed=42, numpy_random_seed=42,
                                 torch_random_seed=42, fewshot_random_seed=42)
    validate_coverage(result)
    result = json.loads(json.dumps(result, default=serializable, ensure_ascii=False, allow_nan=False))
    atomic_json(dest / "results.json", result)
    atomic_json(dest / "summary.json", {"status": "completed", "contract": expected,
        "scores": result["results"], "n_samples": result["n-samples"],
        "results_sha256": sha256(dest / "results.json"),
        "limitations": "Official task families and harness, not a claim of identical published scores. Installed harness/task hashes, dataset documents, native HF backend, seed, FP16 and batch=1 are recorded. No human evaluation or contamination-free claim."})
    verify_result(dest / "summary.json", expected)


if __name__ == "__main__":
    main()
