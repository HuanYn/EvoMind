"""Bounded-memory, stdlib-only integrity audit of a completed EvoMind-V split.

No model or image decoder is loaded. Character-script proportions are heuristics,
not language identification or quality metrics. No exact-match score is produced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import sys
import tempfile
import unicodedata


SPLITS = ("train", "val", "test")
TASKS = ("open", "closed", "ocr")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_LINE_BYTES = 16 * 1024 * 1024
MAX_MESSAGES = 4096


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split(image_hash, seed, val_fraction, test_fraction):
    """Exactly reproduce vision/evomind_v/data.py's floating-point thresholds."""
    value = int(hashlib.sha256(f"{seed}:{image_hash}".encode()).hexdigest(), 16)
    unit = value / (1 << 256)
    return "val" if unit < val_fraction else "test" if unit < val_fraction + test_fraction else "train"


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def character_counts(messages):
    counts = Counter()
    for message in messages:
        # This is a template marker, not naturally occurring English prose.
        for char in message["content"].replace("<image>", ""):
            name = unicodedata.name(char, "")
            if name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
                bucket = "han"
            elif char.isalpha() and "LATIN" in name:
                bucket = "latin"
            elif char.isalpha():
                bucket = "other_letter"
            elif char.isdigit():
                bucket = "digit"
            elif char.isspace():
                bucket = "whitespace"
            else:
                bucket = "other"
            counts[bucket] += 1
    return counts


def validate_messages(messages):
    if not isinstance(messages, list) or not 2 <= len(messages) <= MAX_MESSAGES:
        raise ValueError("conversations must contain 2..4096 messages")
    for message in messages:
        if (not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant")
                or not isinstance(message.get("content"), str)):
            raise ValueError("invalid normalized role/content")
    body = messages[1:] if messages[0]["role"] == "system" else messages
    if len(body) % 2 or any(item["role"] != ("user" if i % 2 == 0 else "assistant")
                            for i, item in enumerate(body)):
        raise ValueError("conversations must alternate complete user/assistant turns")
    if any(not item["content"].strip() for item in body if item["role"] == "assistant"):
        raise ValueError("empty assistant answer")
    if sum(item["content"].count("<image>") for item in messages) != 1:
        raise ValueError("expected exactly one image marker")
    if any("<image>" in item["content"] for item in messages if item["role"] != "user"):
        raise ValueError("image marker outside user turn")
    return len(body) // 2


def safe_image_path(manifest, image_root, value, image_hash):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("invalid image_path")
    windows = PureWindowsPath(value)
    relative = Path(value)
    if relative.is_absolute() or windows.drive or windows.root or ".." in windows.parts:
        raise ValueError("image_path must be a non-traversing relative path")
    resolved = (manifest.parent / relative).resolve()
    if not resolved.is_relative_to(image_root) or resolved == image_root:
        raise ValueError("image_path escapes manifest/images")
    if image_hash not in resolved.name:
        raise ValueError("image filename must include the complete image_hash")
    if not resolved.is_file():
        raise ValueError("image file does not exist")
    return resolved


def _audit(summary_path, manifest, report, work_dir, verify_image_samples, verify_source_hashes):
    errors = Counter()
    examples = []

    def fail(code, detail, line=None):
        errors[code] += 1
        if len(examples) < 50:
            examples.append({"code": code, "line": line, "detail": str(detail)[:500]})

    report["error_counts"] = errors
    report["error_examples"] = examples
    with summary_path.open("rb") as stream:
        summary_bytes = stream.read(MAX_LINE_BYTES + 1)
    if len(summary_bytes) > MAX_LINE_BYTES:
        raise ValueError("Summary exceeds the 16 MiB input memory bound")
    report["summary_sha256"] = hashlib.sha256(summary_bytes).hexdigest()
    summary = json.loads(summary_bytes)
    if not isinstance(summary, dict) or summary.get("status") != "complete" or summary.get("format_version") != 1:
        raise ValueError("Input summary must have status=complete and format_version=1")
    if summary.get("grouping") != "sha256_original_image_bytes":
        raise ValueError("Unsupported or absent image grouping provenance")
    seed, val, test = summary["seed"], summary["val_fraction"], summary.get("test_fraction", 0.0)
    if (not is_int(seed) or not isinstance(val, (int, float)) or isinstance(val, bool)
            or not isinstance(test, (int, float)) or isinstance(test, bool)
            or not math.isfinite(val) or not math.isfinite(test)
            or not 0 < val < 1 or test < 0 or val + test >= 1):
        raise ValueError("Invalid split seed/fractions")
    if not HASH_RE.fullmatch(str(summary.get("manifest_sha256", ""))):
        raise ValueError("Invalid or absent summary manifest_sha256")
    sources = summary.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Summary requires source provenance")
    if not isinstance(summary.get("counts"), dict) or not isinstance(summary.get("unique_images_by_split"), dict):
        raise ValueError("Summary requires counts and unique_images_by_split dictionaries")
    if not is_int(summary.get("unique_images")) or summary["unique_images"] < 0:
        raise ValueError("Invalid summary unique_images")
    report["split_rule"] = {"algorithm": "SHA256(seed:image_hash) / 2**256; val then test then train",
                            "seed": seed, "val_fraction": val, "test_fraction": test}
    report["source_provenance"] = []
    for index, source in enumerate(sources):
        if (not isinstance(source, dict) or not is_int(source.get("total_rows")) or source["total_rows"] < 0
                or not is_int(source.get("size_bytes")) or source["size_bytes"] < 0
                or not HASH_RE.fullmatch(str(source.get("sha256", "")))):
            raise ValueError(f"Invalid source metadata at index {index}")
        path = Path(source["path"])
        if not path.is_absolute():
            raise ValueError(f"Source path must be absolute: {path}")
        item = {"source_index": index, **source, "verification": "existence_and_size_only",
                "content_hash_recomputed": False}
        report["source_provenance"].append(item)
        if not path.is_file():
            fail("missing_source", path)
            continue
        before = path.stat()
        item["observed_size_bytes"] = before.st_size
        if before.st_size != source["size_bytes"]:
            fail("source_size_mismatch", path)
        if verify_source_hashes:
            item["observed_sha256"] = file_hash(path)
            item["content_hash_recomputed"] = True
            item["verification"] = "sha256_and_size"
            if item["observed_sha256"] != source["sha256"]:
                fail("source_hash_mismatch", path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                fail("source_changed_during_audit", path)

    image_root = (manifest.parent / "images").resolve()
    if not image_root.is_relative_to(manifest.parent) or not image_root.is_dir():
        raise ValueError("manifest/images must be an existing directory inside the manifest directory")
    db = sqlite3.connect(str(work_dir / "integrity.sqlite3"))
    try:
        db.execute("PRAGMA cache_size=-8192")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("CREATE TABLE samples (id TEXT PRIMARY KEY, source_index INTEGER, source_row INTEGER, UNIQUE(source_index,source_row))")
        db.execute("CREATE TABLE images (hash TEXT PRIMARY KEY, split TEXT)")
        records, counts, turns, characters, profiles = 0, Counter(), Counter(), Counter(), Counter()
        image_samples = []
        report["image_hash_verification"] = {"requested_limit": verify_image_samples,
            "sampling": "first N unique valid manifest image hashes, deterministic; not random or exhaustive",
            "verified_count": 0, "samples": image_samples}
        digest = hashlib.sha256()
        before = manifest.stat()
        with manifest.open("rb") as stream:
            line_number = 0
            while True:
                raw = stream.readline(MAX_LINE_BYTES + 1)
                if not raw:
                    break
                line_number += 1
                digest.update(raw)
                if len(raw) > MAX_LINE_BYTES:
                    raise ValueError(f"Manifest line {line_number} exceeds {MAX_LINE_BYTES} byte memory bound")
                if not raw.strip():
                    continue
                records += 1
                try:
                    row = json.loads(raw)
                    if not isinstance(row, dict) or row.get("format_version") != 1:
                        raise ValueError("invalid record format_version")
                    sample_id, image_hash, split = row["sample_id"], row["image_hash"], row["split"]
                    if not isinstance(sample_id, str) or not HASH_RE.fullmatch(str(image_hash)) or split not in SPLITS:
                        raise ValueError("invalid sample_id, image_hash, or split")
                    counts[f"{split}_records"] += 1
                    task = row.get("task_type")
                    if task not in TASKS:
                        raise ValueError("invalid task_type; no task inferred from answer contents")
                    counts[f"task_{task}"] += 1
                    counts[f"{split}_task_{task}"] += 1
                    source_index, source_row = row["source_index"], row["source_row"]
                    if (not is_int(source_index) or not 0 <= source_index < len(sources)
                            or not is_int(source_row) or not 0 <= source_row < sources[source_index]["total_rows"]):
                        raise ValueError("source_index/source_row outside declared source range")
                    if sample_id != f"{source_index}:{source_row}:{image_hash[:16]}":
                        fail("sample_id_provenance_mismatch", sample_id, line_number)
                    try:
                        db.execute("INSERT INTO samples VALUES (?,?,?)", (sample_id, source_index, source_row))
                    except sqlite3.IntegrityError:
                        existing = db.execute("SELECT 1 FROM samples WHERE id=?", (sample_id,)).fetchone()
                        fail("duplicate_sample_id" if existing else "duplicate_source_row", sample_id, line_number)
                    previous = db.execute("SELECT split FROM images WHERE hash=?", (image_hash,)).fetchone()
                    if previous is None:
                        db.execute("INSERT INTO images VALUES (?,?)", (image_hash, split))
                    elif previous[0] != split:
                        fail("cross_split_image_leakage", image_hash, line_number)
                    expected = stable_split(image_hash, seed, val, test)
                    if split != expected:
                        fail("deterministic_split_mismatch", f"{sample_id}: {split} != {expected}", line_number)
                    image = safe_image_path(manifest, image_root, row["image_path"], image_hash)
                    if previous is None and len(image_samples) < verify_image_samples:
                        actual = file_hash(image)
                        image_samples.append({"image_hash": image_hash, "observed_sha256": actual,
                                              "image_path": str(image), "matches": actual == image_hash})
                        if actual != image_hash:
                            fail("sample_image_hash_mismatch", image, line_number)
                    messages = row["conversations"]
                    turn_count = validate_messages(messages)
                    answers = row.get("reference_answers")
                    if not isinstance(answers, list) or not answers or any(not isinstance(x, str) or not x.strip() for x in answers):
                        raise ValueError("invalid reference_answers")
                    turns[str(turn_count)] += 1
                    char = character_counts(messages)
                    characters.update(char)
                    profile = ("han_and_latin" if char["han"] and char["latin"] else "han_without_latin"
                               if char["han"] else "latin_without_han" if char["latin"] else "other_or_no_letters")
                    profiles[profile] += 1
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    fail("invalid_record", exc, line_number)
                if records % 1000 == 0:
                    db.commit()
                if records % 10000 == 0:
                    print(json.dumps({"event": "manifest_audit_progress", "records": records,
                                      "error_count": sum(errors.values())}), flush=True)
        db.commit()
        after = manifest.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            fail("manifest_changed_during_audit", manifest)
        report["manifest_sha256"] = digest.hexdigest()
        report["manifest_sha256_expected"] = summary["manifest_sha256"]
        if digest.hexdigest() != summary["manifest_sha256"]:
            fail("manifest_hash_mismatch", manifest)
        unique_images = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        unique_by_split = dict(db.execute("SELECT split, COUNT(*) FROM images GROUP BY split"))
        counts["accepted"] = records
        report["observed_counts"] = dict(counts)
        report["unique_sample_ids"] = db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        report["unique_images"] = unique_images
        report["unique_images_by_split"] = unique_by_split
        expected_counts = summary.get("counts", {})
        for key in ("accepted", *(f"{s}_records" for s in SPLITS), *(f"task_{t}" for t in TASKS)):
            expected_count = expected_counts.get(key, 0)
            if not is_int(expected_count) or expected_count != counts[key]:
                fail("summary_count_mismatch", f"{key}: observed={counts[key]}, summary={expected_count}")
        scanned = expected_counts.get("scanned")
        rejected = [v for k, v in expected_counts.items() if k.startswith("rejected_")]
        if (not is_int(scanned) or any(not is_int(v) or v < 0 for v in rejected)
                or scanned != records + sum(rejected)):
            fail("summary_scanned_rejected_mismatch", "scanned must equal accepted + all declared rejection counts")
        elif scanned > sum(source["total_rows"] for source in sources):
            fail("summary_scanned_out_of_range", scanned)
        max_rows = summary.get("max_rows")
        if max_rows is not None and (not is_int(max_rows) or max_rows <= 0 or scanned > max_rows):
            fail("summary_max_rows_mismatch", max_rows)
        if summary.get("unique_images") != unique_images:
            fail("summary_unique_images_mismatch", unique_images)
        expected_unique = summary.get("unique_images_by_split", {})
        if any(expected_unique.get(s, 0) != unique_by_split.get(s, 0) for s in SPLITS):
            fail("summary_unique_images_by_split_mismatch", unique_by_split)
        if summary.get("official_train_parquet") is not None and summary.get("official_train_rows") != counts["train_records"]:
            fail("summary_official_train_rows_mismatch", summary.get("official_train_rows"))
        if file_hash(summary_path) != report["summary_sha256"]:
            fail("summary_changed_during_audit", summary_path)
        total_characters = sum(characters.values())
        report["character_script_heuristics"] = {
            "policy": "Unicode character-script counts across normalized conversation content; <image> removed. NOT model-detected language; Han is not proof of Chinese. References not counted twice.",
            "character_counts": dict(characters), "total_characters": total_characters,
            "character_proportions": {k: v / total_characters for k, v in characters.items()} if total_characters else {},
            "record_profiles": dict(profiles),
            "record_profile_proportions": {k: v / sum(profiles.values()) for k, v in profiles.items()} if profiles else {}}
        report["assistant_turn_count_distribution"] = dict(turns)
        report["image_hash_verification"]["verified_count"] = len(image_samples)
        report["status"] = "failed" if errors else "pass"
    finally:
        db.close()


def audit_manifest(summary_path, output_path, *, manifest_path=None, verify_image_samples=0, verify_source_hashes=False):
    summary_path, output_path = Path(summary_path).resolve(), Path(output_path).resolve()
    manifest = Path(manifest_path).resolve() if manifest_path else summary_path.parent / "manifest.jsonl"
    if not is_int(verify_image_samples) or not 0 <= verify_image_samples <= 1024:
        raise ValueError("verify_image_samples must be between 0 and 1024")
    if output_path in (summary_path, manifest) or output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing audit or input: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock = output_path.with_name(output_path.name + ".lock")
    lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(lock_fd)
    report = {"audit_version": 1, "status": "failed", "started_at": utc_now(),
        "input_summary": str(summary_path), "input_manifest": str(manifest), "output": str(output_path),
        "runtime": {"python": sys.version, "platform": sys.platform, "pid": os.getpid()},
        "memory_policy": {"sqlite_cache_kib": 8192, "sqlite_temp_store": "FILE",
                          "max_input_line_bytes": MAX_LINE_BYTES, "max_messages_per_record": MAX_MESSAGES,
                          "max_error_examples": 50},
        "auditor_sha256": file_hash(__file__),
        "scope": {"integrity_only": True, "image_bytes_exhaustively_verified": False,
                  "source_content_hashes_requested": verify_source_hashes,
                  "source_rows_redecoded_from_parquet": False,
                  "source_row_binding": "Range and sample_id convention only; no re-decoding or content comparison to source rows",
                  "perceptual_duplicates_checked": False,
                  "open_answer_policy": "No exact-match accuracy for open descriptions; this audit computes no model scores."}}
    try:
        if output_path.exists():
            raise FileExistsError(f"Audit already exists: {output_path}")
        try:
            with tempfile.TemporaryDirectory(prefix="vision_manifest_audit_", dir=output_path.parent) as temporary:
                _audit(summary_path, manifest, report, Path(temporary), verify_image_samples, verify_source_hashes)
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
            report["status"] = "failed"
            report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        report["completed_at"] = utc_now()
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return report
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-summary", required=True, type=Path, help="Completed preparation summary.json")
    parser.add_argument("--manifest", type=Path, help="Default: manifest.jsonl beside input summary")
    parser.add_argument("--output", required=True, type=Path, help="New audit.json; existing output or lock is refused")
    parser.add_argument("--verify-image-samples", type=int, default=0, help="Hash first N unique images (0..1024); default 0 reads no image bytes")
    parser.add_argument("--verify-source-hashes", action="store_true", help="Sequentially rehash original source files; default checks declared hash format, file existence and size only")
    args = parser.parse_args(argv)
    try:
        report = audit_manifest(args.input_summary, args.output, manifest_path=args.manifest,
                                verify_image_samples=args.verify_image_samples, verify_source_hashes=args.verify_source_hashes)
    except (OSError, ValueError) as exc:
        print(f"Audit not started: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "error_counts": report.get("error_counts", {}), "fatal_error": report.get("fatal_error")}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
