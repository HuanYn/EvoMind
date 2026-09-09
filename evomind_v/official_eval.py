"""Read-only identities for the bundled upstream MiniMind-V example test."""
import ast
import hashlib
from pathlib import Path


def examples(root):
    root = Path(root)
    source = root / "eval_vlm.py"
    prompt = None
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prompt" for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                prompt = node.value.value
                break
    if prompt is None or "<image>" not in prompt:
        raise ValueError("Upstream vision description prompt missing")
    paths = sorted(p for p in (root / "dataset/eval_images").iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if len(paths) != 6:
        raise ValueError("Expected the six pinned upstream example images")
    rows = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"sample_id": path.name, "image_path": str(path.resolve()), "image_hash": digest,
                     "split": "official_examples", "task_type": "open", "reference_answers": [],
                     "conversations": [{"role": "user", "content": prompt},
                                       {"role": "assistant", "content": "[not a reference answer; generation-only placeholder]"}]})
    return rows, {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "prompt": prompt, "images": {row["sample_id"]: row["image_hash"] for row in rows},
                  "scope": "Six upstream example images; qualitative only, not heldout accuracy"}
