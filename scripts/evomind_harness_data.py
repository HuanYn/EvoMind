"""Offline transport for pinned Chinese archives; native harness tasks stay intact.

Use the upstream pandas CSV reading semantics, not the normalized local MCQ
records. No remote Python is executed and no prompt/answer normalization occurs.
"""
from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
import re
import zipfile

from evomind_chinese_eval import SOURCES, archive_path, check_archive

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/benchmarks/chinese_public"
ALIASES = {"ceval/ceval-exam": "ceval", "haonan-li/cmmlu": "cmmlu"}


def identity(task):
    name = {"ceval-valid": "ceval", "cmmlu": "cmmlu"}.get(task)
    if name is None:
        return {"transport": "native_huggingface"}
    spec = SOURCES[name]
    path = archive_path(DATA, name, spec)
    check_archive(path, spec)
    return {"transport": "pinned_local_original_csv", "source": spec,
            "archive": str(path), "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_spec_sha256": hashlib.sha256((ROOT / "scripts/evomind_chinese_eval.py").read_bytes()).hexdigest(),
            "normalization": "none; upstream pandas read_csv and datasets feature conversion",
            "remote_dataset_code_executed": False}


def subject_rows(name, subject, split):
    if name not in SOURCES or not re.fullmatch(r"[a-z0-9_]+", subject):
        raise ValueError("Unknown dataset or invalid subject")
    if split not in (("dev", "val", "test") if name == "ceval" else ("dev", "test")):
        raise ValueError("Unknown split")
    import pandas as pd
    member = f"{split}/{subject}_{split}.csv" if name == "ceval" else f"{split}/{subject}.csv"
    with zipfile.ZipFile(archive_path(DATA, name, SOURCES[name])) as archive:
        if archive.namelist().count(member) != 1:
            raise ValueError(f"Missing/duplicate archive member: {member}")
        payload = archive.read(member)
    options = {"header": 0, "index_col": 0} if name == "cmmlu" else {}
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8", **options)
    required = {"Question", "A", "B", "C", "D", "Answer"} if name == "cmmlu" else {"id", "question", "A", "B", "C", "D"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Unexpected source columns: {member}")
    rows = frame.to_dict(orient="records")
    if name == "ceval":
        for row in rows:
            row.setdefault("answer", "")
            row.setdefault("explanation", "")
    if not rows:
        raise ValueError(f"Empty split: {member}")
    return rows


def load_subject(name, subject):
    import datasets
    fields = ({"Question": "string", **{c: "string" for c in "ABCD"}, "Answer": "string"}
              if name == "cmmlu" else {"id": "int32", "question": "string", **{c: "string" for c in "ABCD"},
                                      "answer": "string", "explanation": "string"})
    features = datasets.Features({k: datasets.Value(v) for k, v in fields.items()})
    splits = ("dev", "val", "test") if name == "ceval" else ("dev", "test")
    return datasets.DatasetDict({split: datasets.Dataset.from_list(subject_rows(name, subject, split), features=features)
                                 for split in splits})


@contextmanager
def local_chinese_datasets(task):
    """Intercept only the selected official dataset; restore even on failure."""
    name = {"ceval-valid": "ceval", "cmmlu": "cmmlu"}.get(task)
    if name is None:
        yield
        return
    identity(task)  # Validate exact archive before accepting any task requests.
    import datasets
    original = datasets.load_dataset
    def load(path, name=None, **kwargs):
        selected = ALIASES.get(path)
        expected = {"ceval-valid": "ceval", "cmmlu": "cmmlu"}[task]
        if selected != expected:
            return original(path, name=name, **kwargs)
        if kwargs:
            raise ValueError(f"Unexpected native dataset options: {kwargs}; audit before adapting")
        return load_subject(selected, name)
    datasets.load_dataset = load
    try:
        yield
    finally:
        datasets.load_dataset = original


def preflight():
    """Exercise all real native Chinese task configs, without loading a model."""
    import datasets  # Windows DLL order, before lm_eval imports torch
    from lm_eval.tasks import TaskManager
    from evomind_run import atomic_json, now
    reports = {}
    for task, name in (("ceval-valid", "ceval"), ("cmmlu", "cmmlu")):
        with local_chinese_datasets(task):
            loaded = TaskManager().load(task)
        spec = SOURCES[name]
        counts = {}
        for key, native in loaded["tasks"].items():
            docs = native.eval_docs
            counts[key] = len(docs)
            if len(native.dataset["dev"]) != 5:
                raise ValueError(f"Unexpected few-shot source size: {key}")
            for doc in docs:
                if native.doc_to_target(doc) not in range(4) or native.doc_to_choice(doc) != list("ABCD"):
                    raise ValueError(f"Invalid native answer/choices: {key}")
                if not native.doc_to_text(doc):
                    raise ValueError(f"Empty native prompt: {key}")
        if len(counts) != spec["subjects"] or sum(counts.values()) != spec["rows"]:
            raise ValueError(f"Incomplete Chinese benchmark: {task}")
        reports[task] = {"subjects": len(counts), "rows": sum(counts.values()), "counts": counts,
                         "dataset_loading": identity(task)}
        print(f"{task}: native configs loaded, {len(counts)} subjects, {sum(counts.values())} eval rows; no model run", flush=True)
    report = {"status": "passed", "time": now(), "datasets": reports, "model_evaluated": False}
    atomic_json(ROOT / "artifacts/benchmarks/harness_local_preflight.json", report)
    return report


if __name__ == "__main__":
    preflight()
