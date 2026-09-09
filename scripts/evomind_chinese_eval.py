"""Pinned public C-Eval val / CMMLU test, local zero-shot choice log-likelihood.

This is NOT the official lm_eval or OpenCompass harness protocol. No benchmark
remote code is executed. Preparation uses only the standard library and does not
load torch. Training contamination has NOT been ruled out. Scores are fractions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CHOICES = ("A", "B", "C", "D")
SCHEMA_VERSION = 1
SOURCES = {
    "ceval": {
        "repository": "ceval/ceval-exam",
        "revision": "3923b519fd180e689d0961bf3a032ece929742f3",
        "filename": "ceval-exam.zip", "version": "2023-08-31 CSV archive",
        "sha256": "68786deeea68ff089c56563ee48fab8160da857b77b913437bb504d681fd8e20",
        "bytes": 1548171, "split": "val", "subjects": 52, "rows": 1346,
        "license": "CC-BY-NC-SA-4.0", "member_suffix": "_val.csv",
        "upstream": "https://github.com/hkust-nlp/ceval",
    },
    "cmmlu": {
        "repository": "haonan-li/cmmlu", "resolved_repository": "lmlmcat/cmmlu",
        "revision": "efcc940752ea4a1ea94d2727f11f83858d64fc8e",
        "filename": "cmmlu_v1_0_1.zip", "version": "1.0.1",
        "sha256": "22ecf70b28bef447ee7d8aa5fe144f56996762f901a8537b03b7693773c672a6",
        "bytes": 1078656, "split": "test", "subjects": 67, "rows": 11582,
        "license": "CC-BY-NC-4.0", "member_suffix": ".csv",
        "upstream": "https://github.com/haonan-li/CMMLU",
    },
}
PROTOCOL = {
    "name": "evomind_local_zh_zero_shot_choice_ll_v1",
    "shots": 0, "open_thinking": False, "candidates": list(CHOICES),
    "prompt": "Chinese single-choice instruction, question and four choices; no gold label",
    "template": "tokenizer.apply_chat_template(user, add_generation_prompt=True, open_thinking=False)",
    "assistant_prefix": "答案：",
    "token_boundary": "encode(rendered_prompt + assistant_prefix), then append separately encoded candidate",
    "score": "sum of conditional candidate-token log probabilities; prompt tokens excluded",
    "length_normalization": False, "tie_break": "first in A,B,C,D", "truncation": "forbidden",
    "harness_equivalence": "Not an official lm_eval/OpenCompass harness score",
    "accuracy_units": "fraction in [0,1]", "macro_weighting": "equal weight per subject",
}
LIMITATIONS = [
    "Public benchmark training contamination has not been ruled out; these are not certified clean scores.",
    "The local chat-template/choice-token-boundary protocol is not asserted equivalent to other harnesses.",
    "C-Eval is the pinned 2023 public validation archive, not the later public test release.",
    "An attached medical LoRA is evaluated as that adapter; it is not relabelled as an exam-trained LoRA.",
    "Longer prompts are evaluated without truncation, not asserted to be in the model's training-length distribution.",
]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_url(spec):
    return f"https://huggingface.co/datasets/{spec['repository']}/resolve/{spec['revision']}/{spec['filename']}"


def archive_path(data_dir, name, spec):
    return Path(data_dir) / "sources" / f"{name}-{spec['filename']}"


def check_archive(path, spec):
    if not path.is_file() or path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise ValueError(f"Missing or changed pinned source archive: {path}")


def download_archive(path, spec):
    """Download only an explicitly pinned public archive; never execute dataset code."""
    if path.exists():
        check_archive(path, spec)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".download")
    try:
        request = urllib.request.Request(source_url(spec), headers={"User-Agent": "evomind-public-benchmark/1"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as stream:
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > spec["bytes"]:
                    raise ValueError("Downloaded archive exceeds the pinned source size")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        check_archive(temporary, spec)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_archive(path, name, spec, output=None):
    """Return deterministic provenance while optionally writing normalized JSONL."""
    check_archive(path, spec)
    records_hash = hashlib.sha256()
    members = {}
    identities = set()
    with zipfile.ZipFile(path) as archive:
        selected = sorted(info.filename for info in archive.infolist()
                          if re.fullmatch(re.escape(spec["split"]) + r"/[a-z0-9_]+" + re.escape(spec["member_suffix"]), info.filename))
        if len(selected) != spec["subjects"] or len(set(selected)) != len(selected):
            raise ValueError(f"{name}: expected {spec['subjects']} distinct subject CSVs, got {len(selected)}")
        for member in selected:
            subject = member.split("/", 1)[1].removesuffix(spec["member_suffix"])
            payload = archive.read(member)
            reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
            question_key, answer_key, id_key = ("question", "answer", "id") if name == "ceval" else ("Question", "Answer", "")
            required = {question_key, answer_key, id_key, *CHOICES}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"{member}: unexpected public CSV schema")
            count = 0
            for source_row, raw in enumerate(reader, start=1):
                if None in raw or any(raw.get(key) is None for key in required):
                    raise ValueError(f"{member}:{source_row}: malformed CSV row")
                source_id = raw[id_key].strip()
                question = raw[question_key].strip()
                choices = {letter: raw[letter].strip() for letter in CHOICES}
                answer = raw[answer_key].strip()
                sample_id = f"{name}:{subject}:{source_id}"
                if not source_id or not question or not all(choices.values()) or answer not in CHOICES:
                    raise ValueError(f"{member}:{source_row}: missing question/choice/id or invalid answer")
                if sample_id in identities:
                    raise ValueError(f"Duplicate benchmark identity: {sample_id}")
                identities.add(sample_id)
                record = {"sample_id": sample_id, "dataset": name, "split": spec["split"],
                          "subject": subject, "source_member": member, "source_row": source_row,
                          "source_id": source_id, "question": question, "choices": choices, "answer": answer}
                line = encoded(record) + b"\n"
                records_hash.update(line)
                if output is not None:
                    output.write(line)
                count += 1
            if not count:
                raise ValueError(f"Empty subject CSV: {member}")
            members[subject] = {"member": member, "rows": count, "sha256": hashlib.sha256(payload).hexdigest()}
    rows = sum(item["rows"] for item in members.values())
    if rows != spec["rows"]:
        raise ValueError(f"{name}: expected all {spec['rows']} rows, got {rows}; incomplete data is not a benchmark")
    return {"source": dict(spec, url=source_url(spec)), "rows": rows, "subjects": members,
            "records_file": f"{name}.records.jsonl", "records_sha256": records_hash.hexdigest()}


def validate_prepared(data_dir, sources=None):
    sources = SOURCES if sources is None else sources
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "manifest.json")
    expected = {"schema_version": SCHEMA_VERSION, "datasets": {}}
    for name, spec in sources.items():
        details = normalize_archive(archive_path(data_dir, name, spec), name, spec)
        if sha256(data_dir / details["records_file"]) != details["records_sha256"]:
            raise ValueError(f"Normalized {name} records differ from the pinned source archive")
        expected["datasets"][name] = details
    if manifest != expected:
        raise ValueError("Prepared manifest differs from pinned source provenance/counts/hashes")
    return manifest


def prepare_data(data_dir, sources=None):
    sources = SOURCES if sources is None else sources
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "manifest.json").exists():
        return validate_prepared(data_dir, sources)
    manifest = {"schema_version": SCHEMA_VERSION, "datasets": {}}
    for name, spec in sources.items():
        source = archive_path(data_dir, name, spec)
        download_archive(source, spec)
        destination = data_dir / f"{name}.records.jsonl"
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as stream:
                details = normalize_archive(source, name, spec, stream)
                stream.flush()
                os.fsync(stream.fileno())
            if destination.exists():
                if sha256(destination) != details["records_sha256"]:
                    raise ValueError(f"Refusing to overwrite different prepared records: {destination}")
            else:
                os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        manifest["datasets"][name] = details
    atomic_json(data_dir / "manifest.json", manifest)
    return manifest


def iter_jsonl(path):
    with Path(path).open("rb") as stream:
        for index, line in enumerate(stream, start=1):
            if not line.endswith(b"\n"):
                raise ValueError(f"Incomplete JSONL tail at {path}:{index}; refusing silent data loss")
            if not line.strip():
                raise ValueError(f"Empty JSONL row at {path}:{index}")
            yield json.loads(line)


def question_prompt(record):
    instruction = "以下是单项选择题。请从 A、B、C、D 中选择唯一正确的选项，只回答选项字母。"
    options = "\n".join(f"{letter}. {record['choices'][letter]}" for letter in CHOICES)
    return f"{instruction}\n题目：{record['question']}\n{options}"


def select_choice(scores):
    if set(scores) != set(CHOICES) or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                        or not math.isfinite(v) or v > 1e-6 for v in scores.values()):
        raise ValueError("Candidate scores must be four finite conditional log probabilities")
    return max(CHOICES, key=lambda letter: scores[letter])


class ChoiceScorer:
    def __init__(self, model, tokenizer, *, device="cuda", dtype="bfloat16", max_context=32768):
        import torch
        self.torch, self.model, self.tokenizer = torch, model.eval(), tokenizer
        self.device, self.dtype = device, dtype
        self.max_context = min(max_context, int(model.config.max_position_embeddings))
        self.candidates = {letter: tokenizer.encode(letter, add_special_tokens=False) for letter in CHOICES}
        if any(not tokens for tokens in self.candidates.values()):
            raise ValueError("An answer candidate tokenized to an empty sequence")
        if len({tuple(tokens) for tokens in self.candidates.values()}) != 4:
            raise ValueError("Tokenizer maps distinct answer candidates to the same token sequence")

    def __call__(self, record):
        torch = self.torch
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question_prompt(record)}], tokenize=False,
            add_generation_prompt=True, open_thinking=False)
        context = self.tokenizer.encode(rendered + PROTOCOL["assistant_prefix"], add_special_tokens=False)
        if not context or len(context) + max(map(len, self.candidates.values())) > self.max_context:
            raise ValueError(f"{record['sample_id']}: context length exceeds {self.max_context}; no truncation allowed")
        amp = (lambda: torch.autocast(device_type="cuda", dtype=getattr(torch, self.dtype))) if str(self.device).startswith("cuda") and self.dtype != "float32" else nullcontext
        scores = {}
        with torch.inference_mode(), amp():
            if all(len(tokens) == 1 for tokens in self.candidates.values()):
                # Native HF-compatible MiniMind supports logits_to_keep. Only the
                # final prompt position is needed; avoid a seq*vocab logits buffer.
                ids = torch.tensor([context], dtype=torch.long, device=self.device)
                logits = self.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                scores = {letter: float(log_probs[tokens[0]].item()) for letter, tokens in self.candidates.items()}
            else:
                for letter, candidate in self.candidates.items():
                    # Last input token is unnecessary: each candidate token is
                    # predicted by the preceding position, never by itself.
                    ids = torch.tensor([context + candidate[:-1]], dtype=torch.long, device=self.device)
                    logits = self.model(input_ids=ids, use_cache=False, logits_to_keep=len(candidate)).logits[0].float()
                    log_probs = torch.log_softmax(logits, dim=-1)
                    target = torch.tensor(candidate, dtype=torch.long, device=self.device)
                    scores[letter] = float(log_probs.gather(-1, target[:, None]).sum().item())
        select_choice(scores)
        return {"scores": scores, "context_tokens": len(context),
                "candidate_tokens": {key: len(value) for key, value in self.candidates.items()}}


def load_scorer(args):
    from evomind_load_text_model import load_model
    model, tokenizer = load_model(args.checkpoint, args.lora, device=args.device)
    return ChoiceScorer(model, tokenizer, device=args.device, dtype=args.dtype, max_context=args.max_context)


def make_config(args, manifest):
    def package_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"
    tokenizer = {path.name: sha256(path) for path in sorted((ROOT / "model").glob("*"))
                 if path.name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")}
    if not {"tokenizer.json", "tokenizer_config.json"}.issubset(tokenizer):
        raise ValueError("Required local tokenizer provenance is missing")
    config = {"schema_version": SCHEMA_VERSION, "checkpoint": str(Path(args.checkpoint).resolve()),
              "checkpoint_sha256": sha256(args.checkpoint), "tokenizer": tokenizer,
              "data_dir": str(Path(args.data_dir).resolve()),
              "data_manifest_sha256": sha256(Path(args.data_dir) / "manifest.json"),
              "data_manifest_content_sha256": digest(manifest),
              "datasets": manifest["datasets"], "protocol": PROTOCOL,
              "device": args.device, "dtype": args.dtype, "max_context": args.max_context,
              "precision": "FP32 parameters; BF16 CUDA autocast" if args.device == "cuda" and args.dtype == "bfloat16" else "FP32 parameters and computation",
              "implementation_sha256": sha256(__file__),
              "loader_sha256": sha256(ROOT / "scripts/evomind_load_text_model.py"),
              "model_implementation_sha256": sha256(ROOT / "model/model_minimind.py"),
              "lora_implementation_sha256": sha256(ROOT / "model/model_lora.py"),
              "packages": {name: package_version(name) for name in ("torch", "transformers", "tokenizers")}}
    if args.lora:
        config.update(lora=str(Path(args.lora).resolve()), lora_sha256=sha256(args.lora))
    return config


def prediction_record(record, result, config_hash):
    prediction = select_choice(result["scores"])
    return {"sample_id": record["sample_id"], "subject": record["subject"], "gold": record["answer"],
            "prediction": prediction, "correct": prediction == record["answer"],
            "record_sha256": digest(record), "config_sha256": config_hash, **result}


def validate_prediction(record, prediction, config_hash):
    expected = prediction_record(record, {key: prediction[key] for key in ("scores", "context_tokens", "candidate_tokens")}, config_hash)
    if prediction != expected:
        raise ValueError(f"Prediction identity/score/label/config mismatch: {record['sample_id']}")
    if not isinstance(prediction["context_tokens"], int) or isinstance(prediction["context_tokens"], bool) or prediction["context_tokens"] < 1:
        raise ValueError("Invalid context token count")
    if set(prediction["candidate_tokens"]) != set(CHOICES) or any(type(v) is not int or v < 1 for v in prediction["candidate_tokens"].values()):
        raise ValueError("Invalid candidate token counts")


def inspect_predictions(records_path, predictions_path, config_hash):
    """Validate an exact deterministic prefix before loading the model on resume."""
    if not predictions_path.exists():
        return 0
    predictions = iter_jsonl(predictions_path)
    count = 0
    for record in iter_jsonl(records_path):
        prediction = next(predictions, None)
        if prediction is None:
            return count
        validate_prediction(record, prediction, config_hash)
        count += 1
    if next(predictions, None) is not None:
        raise ValueError("Predictions contain more rows than the complete benchmark")
    return count


def dataset_report(name, details, data_dir, output_dir, config_hash):
    records_path = Path(data_dir) / details["records_file"]
    predictions_path = Path(output_dir) / "predictions" / f"{name}.jsonl"
    count = inspect_predictions(records_path, predictions_path, config_hash)
    if count != details["rows"]:
        raise ValueError(f"{name}: only {count}/{details['rows']} predictions; refusing completed score")
    subjects = defaultdict(lambda: {"rows": 0, "correct": 0})
    for item in iter_jsonl(predictions_path):
        subjects[item["subject"]]["rows"] += 1
        subjects[item["subject"]]["correct"] += int(item["correct"])
    if {key: value["rows"] for key, value in subjects.items()} != {key: value["rows"] for key, value in details["subjects"].items()}:
        raise ValueError("Subject coverage differs from the full pinned split")
    for value in subjects.values():
        value["accuracy"] = value["correct"] / value["rows"]
    correct = sum(value["correct"] for value in subjects.values())
    micro = correct / count
    return {"status": "complete", "split": details["source"]["split"], "rows": count, "correct": correct,
            "accuracy": micro, "micro_accuracy": micro,
            "macro_accuracy": sum(value["accuracy"] for value in subjects.values()) / len(subjects),
            "subjects": dict(sorted(subjects.items())), "source": details["source"],
            "records": {"path": str(records_path.resolve()), "sha256": details["records_sha256"]},
            "records_sha256": details["records_sha256"],
            "predictions": {"path": str(predictions_path.resolve()), "sha256": sha256(predictions_path)},
            "predictions_sha256": sha256(predictions_path)}


def evaluate(args, manifest, *, scorer_factory=load_scorer):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(args, manifest)
    config_hash = digest(config)
    config_path, summary_path = output_dir / "config.json", output_dir / "summary.json"
    if config_path.exists():
        if not args.resume:
            raise ValueError("Evaluation already exists; use --resume for the identical contract")
        if read_json(config_path) != {"config_sha256": config_hash, "config": config}:
            raise ValueError("Resume config/checkpoint/tokenizer/source/implementation differs")
    else:
        if summary_path.exists() or (output_dir / "predictions").exists():
            raise ValueError("Existing results lack a matching config; refusing overwrite")
        atomic_json(config_path, {"config_sha256": config_hash, "config": config})
    if summary_path.exists() and read_json(summary_path).get("status") == "completed":
        previous = read_json(summary_path)
        reports = {name: dataset_report(name, details, args.data_dir, output_dir, config_hash)
                   for name, details in manifest["datasets"].items()}
        if (previous.get("config_sha256") != config_hash or previous.get("datasets") != reports
                or previous.get("checkpoint_sha256") != config["checkpoint_sha256"]
                or previous.get("lora_sha256") != config.get("lora_sha256")
                or previous.get("protocol") != PROTOCOL or previous.get("limitations") != LIMITATIONS):
            raise ValueError("Completed summary or prediction hashes changed")
        print("Validated completed Chinese evaluation; no model loaded.", flush=True)
        return previous
    started = now()
    scorer = None
    try:
        prefixes = {name: inspect_predictions(Path(args.data_dir) / details["records_file"],
                                              output_dir / "predictions" / f"{name}.jsonl", config_hash)
                    for name, details in manifest["datasets"].items()}
        (output_dir / "predictions").mkdir(exist_ok=True)
        for name, details in manifest["datasets"].items():
            records_path = Path(args.data_dir) / details["records_file"]
            predictions_path = output_dir / "predictions" / f"{name}.jsonl"
            if prefixes[name] == details["rows"]:
                continue
            if scorer is None:
                scorer = scorer_factory(args)
            with predictions_path.open("ab") as stream:
                for index, record in enumerate(iter_jsonl(records_path)):
                    if index < prefixes[name]:
                        continue
                    result = prediction_record(record, scorer(record), config_hash)
                    validate_prediction(record, result, config_hash)
                    stream.write(encoded(result) + b"\n")
                    stream.flush()
                    if (index + 1) % 25 == 0:
                        os.fsync(stream.fileno())
                        print(json.dumps({"dataset": name, "completed_rows": index + 1, "total_rows": details["rows"]}), flush=True)
                os.fsync(stream.fileno())
        reports = {name: dataset_report(name, details, args.data_dir, output_dir, config_hash)
                   for name, details in manifest["datasets"].items()}
        summary = {"status": "completed", "schema_version": SCHEMA_VERSION, "config_sha256": config_hash,
                   "checkpoint_sha256": config["checkpoint_sha256"], "datasets": reports,
                   "protocol": PROTOCOL, "limitations": LIMITATIONS, "started_at": started, "completed_at": now()}
        if args.lora:
            summary["lora_sha256"] = config["lora_sha256"]
            summary["adapter_path"] = config["lora"]
        atomic_json(summary_path, summary)
        return summary
    except BaseException as error:
        atomic_json(summary_path, {"status": "failed", "config_sha256": config_hash,
                                  "checkpoint_sha256": config["checkpoint_sha256"], "started_at": started,
                                  "failed_at": now(), "error": f"{type(error).__name__}: {error}",
                                  "completion_claim": False, "limitations": LIMITATIONS})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--lora", type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/benchmarks/chinese_public")
    result.add_argument("--prepare-only", action="store_true", help="Pinned public data only; no torch/model/GPU import")
    result.add_argument("--resume", action="store_true", help="Require identical config; validate and reuse exact prediction prefix")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    result.add_argument("--max-context", type=int, default=32768, help="Hard failure above model/context bound; never truncates or drops rows")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if not arguments.prepare_only and not arguments.checkpoint:
        parser().error("--checkpoint is required unless --prepare-only")
    if arguments.max_context < 2:
        parser().error("--max-context must be at least 2")
    if arguments.prepare_only and (arguments.checkpoint or arguments.lora):
        parser().error("--prepare-only does not accept checkpoint/adapter evaluation inputs")
    from evomind_run import exclusive_run_lock
    arguments.output_dir = arguments.output_dir.resolve()
    arguments.data_dir = arguments.data_dir.resolve()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if arguments.prepare_only:
            arguments.data_dir.mkdir(parents=True, exist_ok=True)
            with exclusive_run_lock(arguments.data_dir):
                manifest = prepare_data(arguments.data_dir)
            receipt = {"status": "prepared", "data_dir": str(arguments.data_dir),
                       "manifest_sha256": sha256(arguments.data_dir / "manifest.json"),
                       "datasets": manifest["datasets"], "evaluation_executed": False}
            with exclusive_run_lock(arguments.output_dir):
                path = arguments.output_dir / "preparation.json"
                if path.exists() and read_json(path) != receipt:
                    raise ValueError("Refusing to overwrite a different preparation receipt")
                atomic_json(path, receipt)
            print(json.dumps({"status": "prepared", "rows": {name: spec["rows"] for name, spec in SOURCES.items()}}, ensure_ascii=False), flush=True)
        else:
            manifest = validate_prepared(arguments.data_dir)
            with exclusive_run_lock(arguments.output_dir):
                result = evaluate(arguments, manifest)
            print(json.dumps({"status": result["status"], "summary": str(arguments.output_dir / "summary.json")}), flush=True)
        return 0
    except Exception as error:
        print(f"Chinese evaluation failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
