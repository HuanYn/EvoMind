"""Deterministically reorder a complete vision manifest with bounded memory.

This standard-library-only preparation step does not import ML libraries, read
image pixels, start training, or alter the official source parquet. Records are
ordered by SHA256(UTF8(f"{seed}:{sample_id}")), then by sample_id. A disk-backed
SQLite B-tree provides that order without loading all rows or sort keys in RAM.
Original image-group train/val/test assignments and all semantic fields survive.
Only image_path is rewritten relative to the new manifest's final directory.

Example, after source preparation has completed:
  python scripts/evomind_shuffle_manifest.py --input vision/dataset/evomind_split/manifest.jsonl \
    --output-dir vision/dataset/evomind_controlled --seed 42

Random order removes source-order prefix selection; it does not balance
languages/tasks or certify the quality of a particular 20k-record subset.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time


SCHEMA_VERSION = 1
METHOD = "sha256-seed-sample-id-sqlite-btree-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train", "val", "test"}
ORDER_QUERY = "SELECT body, split FROM records ORDER BY order_key, sample_id"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def order_key(seed, sample_id):
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()


@contextmanager
def exclusive_output_lock(path):
    """An OS lock blocks duplicate preparation, and releases after process exit."""
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another shuffle owns {path}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _normalize_record(raw, source_parent, output_dir, line_number):
    row = json.loads(raw, parse_constant=_reject_constant)
    if not isinstance(row, dict) or row.get("format_version") != 1:
        raise ValueError(f"Line {line_number}: expected a format_version=1 manifest record")
    if not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
        raise ValueError(f"Line {line_number}: sample_id must be a nonempty string")
    if row.get("split") not in SPLITS:
        raise ValueError(f"Line {line_number}: split must be train, val or test")
    if not isinstance(row.get("image_hash"), str) or re.fullmatch(r"[0-9a-f]{64}", row["image_hash"]) is None:
        raise ValueError(f"Line {line_number}: image_hash must be a lowercase SHA256 digest")
    if not isinstance(row.get("image_path"), str) or not row["image_path"]:
        raise ValueError(f"Line {line_number}: image_path must be a nonempty string")
    original = Path(row["image_path"])
    resolved = (original if original.is_absolute() else source_parent / original).resolve()
    try:
        row["image_path"] = Path(os.path.relpath(resolved, output_dir)).as_posix()
    except ValueError as exc:
        raise ValueError("Images and output manifest must share a filesystem volume for relative paths") from exc
    body = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return row, body


def _same_file_snapshot(left, right):
    return (left.st_size, left.st_mtime_ns, left.st_ino) == (right.st_size, right.st_mtime_ns, right.st_ino)


def _verify_existing(source, output_dir, seed, expected_source_sha256):
    summary_path, manifest_path = output_dir / "summary.json", output_dir / "manifest.jsonl"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileExistsError(f"Output is nonempty without a complete manifest/summary pair: {output_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("schema_version"), summary.get("status"), summary.get("method"), summary.get("seed")) != (
            SCHEMA_VERSION, "complete", METHOD, seed):
        raise ValueError("Existing output has a different shuffle definition or is incomplete")
    if Path(summary.get("source_manifest", "")).resolve() != source:
        raise ValueError("Existing shuffle belongs to another source manifest")
    before = source.stat()
    actual_source = sha256_file(source)
    if not _same_file_snapshot(before, source.stat()):
        raise RuntimeError("Source manifest changed during verification; wait for preparation to finish")
    if actual_source != summary.get("source_manifest_sha256"):
        raise ValueError("Source manifest changed since the existing shuffle was built")
    if summary.get("input_manifest_sha256") != actual_source:
        raise ValueError("Existing summary input hash fields disagree")
    if expected_source_sha256 is not None and actual_source != expected_source_sha256:
        raise ValueError("Source hash differs from --expected-source-sha256")
    actual_output = sha256_file(manifest_path)
    if actual_output != summary.get("manifest_sha256"):
        raise ValueError("Existing shuffled manifest hash differs from its summary")
    if summary.get("output_manifest_sha256") != actual_output:
        raise ValueError("Existing summary output hash fields disagree")
    return summary


def shuffle_manifest(source_manifest, output_dir, *, seed=42, sqlite_cache_mib=32,
                     max_line_mib=16, commit_every=10000, progress_every=100000,
                     expected_source_sha256=None):
    """Publish a manifest and its summary together via an atomic directory rename.

    Work is limited to one JSON record plus SQLite's configured page cache;
    Python never stores a collection proportional to the number of source rows.
    Duplicate sample IDs and cross-split occurrences of the same image hash are
    rejected using disk-backed indexes. Incomplete work is never published.
    Completed matching outputs are verified and reused, without rewriting them.
    """
    requested_output = Path(output_dir)
    if requested_output.is_symlink():
        raise ValueError("Output directory cannot be a symlink")
    source, output_dir = Path(source_manifest).resolve(), requested_output.resolve()
    if output_dir == PROJECT_ROOT or not output_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("Output must be a dedicated directory inside the evomind project")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if sqlite_cache_mib < 1 or max_line_mib < 1 or commit_every < 1 or progress_every < 0:
        raise ValueError("Cache/line/commit limits must be positive; progress interval must be nonnegative")
    if expected_source_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256) is None:
        raise ValueError("expected_source_sha256 must be a lowercase SHA256 digest")
    if not source.is_file():
        raise FileNotFoundError(f"Source manifest is not complete/available yet: {source}")
    if source.is_relative_to(output_dir):
        raise ValueError("Output directory must not contain the source manifest")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.shuffle.lock"
    with exclusive_output_lock(lock_path):
        if output_dir.exists():
            if not output_dir.is_dir():
                raise FileExistsError(f"Output target is not a directory: {output_dir}")
            if any(output_dir.iterdir()):
                return _verify_existing(source, output_dir, seed, expected_source_sha256)
            output_dir.rmdir()  # Empty directory only; no existing material is removed.
        initial_stat = source.stat()
        # The work DB and published JSON coexist until completion. This is a
        # conservative estimate, not a promise about arbitrary JSON expansion.
        estimate = 3 * initial_stat.st_size + 64 * 1024 * 1024
        if shutil.disk_usage(output_dir.parent).free < estimate:
            raise OSError(f"Insufficient disk space for estimated {estimate} bytes of shuffle work")
        started = time.perf_counter()
        source_digest = hashlib.sha256()
        counts, blank_lines, rows = Counter(), 0, 0
        maximum_line_bytes = int(max_line_mib * 1024 * 1024)
        with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-shuffle-", dir=output_dir.parent) as temporary:
            work = Path(temporary).resolve()
            if work.parent != output_dir.parent or not work.name.startswith(f".{output_dir.name}-shuffle-"):
                raise RuntimeError("Temporary shuffle workspace escaped the selected output parent")
            publish = work / "publish"
            publish.mkdir()
            database_path = work / "records.sqlite3"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(f"PRAGMA cache_size=-{int(sqlite_cache_mib * 1024)}")
                connection.execute("PRAGMA mmap_size=0")
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute("PRAGMA locking_mode=EXCLUSIVE")
                # The disposable work DB needs no crash recovery. Only the
                # fsynced final files are published, after the connection closes.
                connection.execute("CREATE TABLE records (order_key BLOB NOT NULL, sample_id TEXT NOT NULL, body TEXT NOT NULL, split TEXT NOT NULL, PRIMARY KEY(order_key, sample_id)) WITHOUT ROWID")
                connection.execute("CREATE UNIQUE INDEX unique_sample_id ON records(sample_id)")
                connection.execute("CREATE TABLE image_splits (image_hash TEXT PRIMARY KEY, split TEXT NOT NULL CHECK(split IN ('train','val','test'))) WITHOUT ROWID")
                with source.open("rb") as stream:
                    line_number = 0
                    while True:
                        raw = stream.readline(maximum_line_bytes + 1)
                        if not raw:
                            break
                        line_number += 1
                        if len(raw) > maximum_line_bytes:
                            raise ValueError(f"Line {line_number} exceeds --max-line-mib; no output published")
                        source_digest.update(raw)
                        if not raw.strip():
                            blank_lines += 1
                            continue
                        row, body = _normalize_record(raw, source.parent, output_dir, line_number)
                        try:
                            connection.execute(
                                "INSERT INTO image_splits(image_hash, split) VALUES (?, ?) ON CONFLICT(image_hash) DO UPDATE SET split=CASE WHEN image_splits.split=excluded.split THEN image_splits.split ELSE NULL END",
                                (row["image_hash"], row["split"]))
                            connection.execute("INSERT INTO records VALUES (?, ?, ?, ?)",
                                               (order_key(seed, row["sample_id"]), row["sample_id"], body, row["split"]))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(f"Line {line_number}: duplicate sample_id or the same image_hash appears in multiple splits") from exc
                        rows += 1
                        counts[row["split"]] += 1
                        if rows % commit_every == 0:
                            connection.commit()
                        if progress_every and rows % progress_every == 0:
                            print(json.dumps({"event": "shuffle_ingest", "records": rows,
                                              "source_bytes_read": stream.tell(), "source_bytes": initial_stat.st_size,
                                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
                connection.commit()
                if rows == 0:
                    raise ValueError("Source manifest contains no records")
                if not _same_file_snapshot(initial_stat, source.stat()):
                    raise RuntimeError("Source manifest changed while reading; wait for source preparation to finish")
                source_hash = source_digest.hexdigest()
                if expected_source_sha256 is not None and source_hash != expected_source_sha256:
                    raise ValueError("Source hash differs from --expected-source-sha256")
                plan = [row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + ORDER_QUERY)]
                if any("TEMP B-TREE" in detail.upper() for detail in plan):
                    raise RuntimeError("SQLite attempted a separate sort instead of scanning the ordered disk B-tree")
                output_digest, output_counts, written = hashlib.sha256(), Counter(), 0
                destination = publish / "manifest.jsonl"
                with destination.open("xb") as stream:
                    for body, split in connection.execute(ORDER_QUERY):
                        raw = (body + "\n").encode("utf-8")
                        stream.write(raw)
                        output_digest.update(raw)
                        output_counts[split] += 1
                        written += 1
                    stream.flush()
                    os.fsync(stream.fileno())
                if written != rows or output_counts != counts:
                    raise RuntimeError("Record count or split counts changed during shuffle")
                unique_images = connection.execute("SELECT COUNT(*) FROM image_splits").fetchone()[0]
                summary = {"schema_version": SCHEMA_VERSION, "status": "complete", "method": METHOD,
                           "seed": seed, "source_manifest": str(source), "source_manifest_sha256": source_hash,
                           "manifest": str(output_dir / "manifest.jsonl"), "manifest_sha256": output_digest.hexdigest(),
                           "input_manifest_sha256": source_hash, "output_manifest_sha256": output_digest.hexdigest(),
                           "records": rows, "unique_images": unique_images, "split_counts": dict(sorted(counts.items())),
                           "counts": {"records": rows, "unique_images": unique_images,
                                      "by_split": dict(sorted(counts.items()))},
                           "source_bytes": initial_stat.st_size, "manifest_bytes": destination.stat().st_size,
                           "skipped_blank_lines": blank_lines, "sqlite_version": sqlite3.sqlite_version,
                           "sqlite_cache_mib": sqlite_cache_mib, "max_line_mib": max_line_mib,
                           "commit_every": commit_every, "export_query_plan": plan,
                           "ordering": "ascending raw SHA256 bytes of UTF8(f'{seed}:{sample_id}'), then sample_id BINARY",
                           "preservation": "All records and fields preserved except image_path rewritten relative to output directory; original image-group split unchanged; official parquet untouched",
                           "memory_policy": "One JSON record at a time; fixed SQLite page cache; mmap disabled; rows and indexes on disk; no Python collection grows with source row count",
                           "image_integrity_scope": "Paths are resolved and preserved; image bytes are not reread here. Training/evaluation verifies image_hash when loading each selected image.",
                           "selection_limit": "Random ordering removes source-prefix bias; it is not language/task balancing or a guarantee of Chinese-only samples.",
                           "elapsed_seconds": time.perf_counter() - started}
            finally:
                connection.close()
            if not _same_file_snapshot(initial_stat, source.stat()):
                raise RuntimeError("Source manifest changed before publication")
            with (publish / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic directory publication prevents a final manifest without its
            # matching summary. The work database remains outside the final dir.
            publish.rename(output_dir)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--source-manifest", dest="source_manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sqlite-cache-mib", type=int, default=32)
    parser.add_argument("--max-line-mib", type=int, default=16)
    parser.add_argument("--commit-every", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    result = shuffle_manifest(**vars(args))
    print(json.dumps({"event": "shuffle_complete", "status": result["status"], "records": result["records"],
                      "seed": result["seed"], "source_manifest_sha256": result["source_manifest_sha256"],
                      "manifest_sha256": result["manifest_sha256"], "manifest": result["manifest"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
