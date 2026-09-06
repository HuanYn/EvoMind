"""Versioned JSONL training-history utilities.

The module intentionally depends only on the Python standard library so that
training can record metrics without importing a plotting stack.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4


SCHEMA_VERSION = 1
RECORD_TYPE = "metrics"
ARCHITECTURES = frozenset({"dense", "moe"})


class TrainingHistoryError(ValueError):
    """Raised when a history file violates the JSONL v1 contract."""


def _sanitise_value(value: Any, path: str, invalid: list[str]) -> Any:
    """Return a strict-JSON-compatible value and record non-finite paths."""
    if isinstance(value, float):
        if not math.isfinite(value):
            invalid.append(path)
            return None
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _sanitise_value(
                item,
                f"{path}.{key}" if path else str(key),
                invalid,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _sanitise_value(item, f"{path}[{index}]", invalid)
            for index, item in enumerate(value)
        ]
    return value


def sanitise_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a record, replacing NaN/Inf with ``None``.

    Every replaced value is named in ``invalid_metrics``. Missing metrics and
    deliberately supplied ``None`` values are therefore distinguishable from
    metrics that became invalid during training.
    """
    invalid: list[str] = []
    clean = _sanitise_value(dict(record), "", invalid)
    previous = clean.get("invalid_metrics", [])
    if previous is None:
        previous = []
    if not isinstance(previous, list) or not all(
        isinstance(name, str) for name in previous
    ):
        raise TrainingHistoryError("invalid_metrics must be a list of strings")
    clean["invalid_metrics"] = sorted(set(previous).union(invalid))
    return clean


def _validate_record(record: Mapping[str, Any], *, location: str = "record") -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise TrainingHistoryError(
            f"{location}: schema_version must be {SCHEMA_VERSION}"
        )
    if record.get("record_type") != RECORD_TYPE:
        raise TrainingHistoryError(
            f"{location}: record_type must be {RECORD_TYPE!r}"
        )

    for field in ("run_id", "stage"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise TrainingHistoryError(f"{location}: {field} must be a non-empty string")

    architecture = record.get("architecture")
    if architecture not in ARCHITECTURES:
        allowed = ", ".join(sorted(ARCHITECTURES))
        raise TrainingHistoryError(
            f"{location}: architecture must be one of: {allowed}"
        )

    step = record.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise TrainingHistoryError(f"{location}: step must be a non-negative integer")

    invalid = record.get("invalid_metrics")
    if not isinstance(invalid, list) or not all(isinstance(name, str) for name in invalid):
        raise TrainingHistoryError(
            f"{location}: invalid_metrics must be a list of strings"
        )


def make_record(
    *,
    run_id: str,
    stage: str,
    architecture: str,
    step: int,
    **metrics: Any,
) -> dict[str, Any]:
    """Build, sanitise and validate one JSONL v1 optimizer-step record."""
    reserved = {
        "schema_version",
        "record_type",
        "run_id",
        "stage",
        "architecture",
        "step",
    }
    overlap = reserved.intersection(metrics)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise TrainingHistoryError(f"reserved fields supplied as metrics: {names}")
    record = sanitise_record(
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": RECORD_TYPE,
            "run_id": run_id,
            "stage": stage,
            "architecture": architecture,
            "step": step,
            **metrics,
        }
    )
    _validate_record(record)
    return record


def _encode_record(record: Mapping[str, Any]) -> str:
    clean = sanitise_record(record)
    _validate_record(clean)
    try:
        return json.dumps(
            clean,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TrainingHistoryError(f"record is not JSON serialisable: {exc}") from exc


def append_record(path: str | Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Append one complete UTF-8 JSON line and return its sanitised form."""
    path = Path(path)
    clean = sanitise_record(record)
    line = _encode_record(clean)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.write("\n")
        handle.flush()
    return clean


def load_history(
    path: str | Path,
    *,
    recover_truncated_tail: bool = True,
) -> list[dict[str, Any]]:
    """Load JSONL v1 records.

    A malformed *unterminated final line* is treated as a crash-truncated write
    and ignored. Any malformed complete line, including a bad middle line,
    raises :class:`TrainingHistoryError`.
    """
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    identity: tuple[str, str, str] | None = None

    for line_index, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError) as exc:
            is_last = line_index == len(lines)
            is_unterminated = not raw_line.endswith(("\n", "\r"))
            if recover_truncated_tail and is_last and is_unterminated:
                warnings.warn(
                    f"Ignoring truncated final line in {path} (line {line_index})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            raise TrainingHistoryError(
                f"{path}:{line_index}: malformed JSONL record: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise TrainingHistoryError(
                f"{path}:{line_index}: each JSONL line must contain an object"
            )
        clean = sanitise_record(payload)
        _validate_record(clean, location=f"{path}:{line_index}")
        current_identity = (
            clean["run_id"],
            clean["stage"],
            clean["architecture"],
        )
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise TrainingHistoryError(
                f"{path}:{line_index}: run_id/stage/architecture changed within one file"
            )
        records.append(clean)
    return records


def deduplicate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    max_step: int | None = None,
) -> list[dict[str, Any]]:
    """Keep the last record for each step, optionally trimming future steps."""
    if max_step is not None and (
        isinstance(max_step, bool) or not isinstance(max_step, int) or max_step < 0
    ):
        raise TrainingHistoryError("max_step must be a non-negative integer")
    by_step: dict[int, dict[str, Any]] = {}
    identity: tuple[str, str, str] | None = None
    for index, record in enumerate(records, start=1):
        clean = sanitise_record(record)
        _validate_record(clean, location=f"record {index}")
        current_identity = (
            clean["run_id"],
            clean["stage"],
            clean["architecture"],
        )
        if identity is None:
            identity = current_identity
        elif identity != current_identity:
            raise TrainingHistoryError(
                "cannot combine records with different run_id/stage/architecture"
            )
        if max_step is None or clean["step"] <= max_step:
            by_step[clean["step"]] = clean
    return [by_step[step] for step in sorted(by_step)]


def _atomic_rewrite(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(_encode_record(record))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def prepare_history_for_resume(
    path: str | Path,
    checkpoint_step: int,
) -> list[dict[str, Any]]:
    """Align a history file to a resumed checkpoint.

    Records beyond ``checkpoint_step`` are removed and duplicate steps collapse
    to their last occurrence. The repaired file is rewritten atomically.
    """
    path = Path(path)
    records = load_history(path)
    repaired = deduplicate_records(records, max_step=checkpoint_step)
    if path.exists() or repaired:
        _atomic_rewrite(path, repaired)
    return repaired


class TrainingHistoryWriter:
    """Stateful writer enforcing exactly one increasing record per step."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        stage: str,
        architecture: str,
        checkpoint_step: int | None = None,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.stage = stage
        self.architecture = architecture
        make_record(
            run_id=run_id,
            stage=stage,
            architecture=architecture,
            step=0,
        )

        if checkpoint_step is not None and overwrite:
            raise TrainingHistoryError(
                "overwrite=True cannot be combined with checkpoint_step resume"
            )

        if checkpoint_step is None:
            is_non_empty = self.path.exists() and self.path.stat().st_size > 0
            if is_non_empty and not overwrite:
                raise TrainingHistoryError(
                    "new training found a non-empty history; pass overwrite=True "
                    "to atomically start a new log, or provide checkpoint_step to resume"
                )
            if overwrite:
                _atomic_rewrite(self.path, [])
            existing: list[dict[str, Any]] = []
        else:
            existing = prepare_history_for_resume(self.path, checkpoint_step)

        for record in existing:
            identity = (
                record["run_id"],
                record["stage"],
                record["architecture"],
            )
            if identity != (run_id, stage, architecture):
                raise TrainingHistoryError(
                    "writer identity does not match the existing history file"
                )
        self.last_step = existing[-1]["step"] if existing else -1

    def append(self, step: int, **metrics: Any) -> dict[str, Any]:
        if step <= self.last_step:
            raise TrainingHistoryError(
                f"step {step} is not newer than the last recorded step {self.last_step}"
            )
        record = make_record(
            run_id=self.run_id,
            stage=self.stage,
            architecture=self.architecture,
            step=step,
            **metrics,
        )
        append_record(self.path, record)
        self.last_step = step
        return record


# Explicit aliases make the public API easy to discover at call sites.
append_training_record = append_record
load_training_history = load_history


__all__ = [
    "ARCHITECTURES",
    "RECORD_TYPE",
    "SCHEMA_VERSION",
    "TrainingHistoryError",
    "TrainingHistoryWriter",
    "append_record",
    "append_training_record",
    "deduplicate_records",
    "load_history",
    "load_training_history",
    "make_record",
    "prepare_history_for_resume",
    "sanitise_record",
]
