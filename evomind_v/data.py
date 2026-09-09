"""Deterministic single-image data for the A/B/C MiniMind-V experiments.

No sample packing or token truncation is performed. All variants use the same
five-view eligibility check; an overlength conversation is rejected in full.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_TOKEN = "<|image_pad|>"
TOKENS_PER_VIEW = 64
VIEW_NAMES = ("global", "top-left", "top-right", "bottom-left", "bottom-right")
ASSISTANT_PREFIX = "<think>\n\n</think>\n\n"
FORMAT_VERSION = 1


class SampleRejected(ValueError):
    """An explicit, countable dataset rejection, never a truncated answer."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_split(image_hash: str, seed: int, val_fraction: float, test_fraction: float = 0.0) -> str:
    if not 0 < val_fraction < 1 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Require 0 < val_fraction, 0 <= test_fraction, and val + test < 1")
    value = int(hashlib.sha256(f"{seed}:{image_hash}".encode()).hexdigest(), 16)
    unit = value / (1 << 256)
    return "val" if unit < val_fraction else "test" if unit < val_fraction + test_fraction else "train"


def normalize_conversations(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SampleRejected("invalid_conversations_json") from exc
    if not isinstance(value, list) or len(value) < 2:
        raise SampleRejected("invalid_conversations")
    messages = []
    for turn in value:
        if not isinstance(turn, dict):
            raise SampleRejected("invalid_turn")
        if any(turn.get(k) for k in ("tools", "functions", "tool_calls", "reasoning_content")):
            raise SampleRejected("unsupported_tools_or_reasoning")
        role = turn.get("role", turn.get("from"))
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        content = turn.get("content", turn.get("value"))
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            raise SampleRejected("invalid_role_or_content")
        if any(token in content for token in ("<|im_start|>", "<|im_end|>", IMAGE_TOKEN)):
            raise SampleRejected("reserved_token_in_content")
        if role == "assistant" and content.startswith(ASSISTANT_PREFIX):
            content = content[len(ASSISTANT_PREFIX):]
        if role == "assistant" and ("<think>" in content or "</think>" in content):
            raise SampleRejected("unsupported_nonempty_reasoning")
        if role == "assistant" and not content.strip():
            raise SampleRejected("empty_answer")
        if role != "user" and "<image>" in content:
            raise SampleRejected("image_outside_user_turn")
        messages.append({"role": role, "content": content})
    body = messages[1:] if messages[0]["role"] == "system" else messages
    if not body or len(body) % 2 or any(
        turn["role"] != ("user" if index % 2 == 0 else "assistant")
        for index, turn in enumerate(body)
    ):
        raise SampleRejected("nonalternating_or_unfinished_chat")
    markers = sum(message["content"].count("<image>") for message in messages)
    if markers != 1:
        raise SampleRejected("expected_one_original_image_marker")
    return messages


def extract_image_bytes(value: Any) -> bytes:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise SampleRejected("multiple_or_missing_original_images")
        value = value[0]
    if not isinstance(value, (bytes, bytearray, memoryview)) or not value:
        raise SampleRejected("invalid_image_bytes")
    return bytes(value)


def prepare_parquet(parquet_paths, output_dir, *, seed=42, val_fraction=0.1,
                    max_rows=None, annotations_path=None, official_train_parquet=None, test_fraction=0.0) -> dict:
    """Stream the official conversations/image_bytes schema to an auditable manifest.

    The hash is over original image bytes. This groups byte-identical images;
    visually identical re-encodings require a separate perceptual-duplicate audit.
    Optional annotations are JSONL rows keyed by sample_id, with task_type
    (open/closed/ocr) and reference_answers. Unknown annotation IDs are rejected.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    stable_split("validation", seed, val_fraction, test_fraction)
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    if manifest.exists() or summary_path.exists():
        raise FileExistsError(f"Use a new output directory; manifest already exists in {output_dir}")
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    annotations = {}
    if annotations_path:
        for line in Path(annotations_path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["sample_id"] in annotations:
                raise ValueError("duplicate annotation sample_id")
            annotations[row["sample_id"]] = row
    used_annotations = set()
    stats = Counter()
    image_splits = {}
    sources = []
    official_writer = None
    official_target = Path(official_train_parquet).resolve() if official_train_parquet else None
    official_tmp = official_target.with_suffix(".parquet.tmp") if official_target else None
    if official_target:
        if official_target.exists() or official_tmp.exists():
            raise FileExistsError(f"Official train export target already exists: {official_target}")
        official_target.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_suffix(".jsonl.tmp")
    with ExitStack() as stack:
        stream = stack.enter_context(tmp.open("x", encoding="utf-8", newline="\n"))
        for source_index, source in enumerate(parquet_paths):
            source = Path(source).resolve()
            parquet = pq.ParquetFile(source)
            if not {"conversations", "image_bytes"}.issubset(parquet.schema_arrow.names):
                raise ValueError(f"{source}: requires conversations and image_bytes columns")
            sources.append({"path": str(source), "size_bytes": source.stat().st_size,
                            "total_rows": parquet.metadata.num_rows, "sha256": sha256_file(source)})
            print(json.dumps({"event": "parquet_metadata", **sources[-1]}, ensure_ascii=False), flush=True)
            if official_target:
                if official_writer is None:
                    official_writer = stack.enter_context(pq.ParquetWriter(official_tmp, parquet.schema_arrow, compression="zstd"))
                elif not official_writer.schema.equals(parquet.schema_arrow):
                    raise ValueError("All source schemas must match for one official train parquet")
            columns = [c for c in ("conversations", "image_bytes", "task_type", "reference_answers")
                       if c in parquet.schema_arrow.names]
            source_row = 0
            for batch in parquet.iter_batches(batch_size=32, columns=None if official_target else columns):
                official_indices = []
                for batch_index, raw in enumerate(batch.to_pylist()):
                    if max_rows is not None and stats["scanned"] >= max_rows:
                        break
                    row_index = source_row
                    source_row += 1
                    stats["scanned"] += 1
                    try:
                        messages = normalize_conversations(raw["conversations"])
                        image_bytes = extract_image_bytes(raw["image_bytes"])
                        try:
                            with Image.open(io.BytesIO(image_bytes)) as image:
                                image_format = image.format
                                image.verify()
                        except (OSError, Image.DecompressionBombError) as exc:
                            raise SampleRejected("invalid_image") from exc
                        image_hash = hashlib.sha256(image_bytes).hexdigest()
                        sample_id = f"{source_index}:{row_index}:{image_hash[:16]}"
                        annotation = annotations.get(sample_id, {})
                        task_type = annotation.get("task_type", raw.get("task_type")) or "open"
                        if task_type not in ("open", "closed", "ocr"):
                            raise SampleRejected("invalid_task_type")
                        answers = annotation.get("reference_answers", raw.get("reference_answers"))
                        if answers is None:
                            answers = [messages[-1]["content"]]
                        if isinstance(answers, str):
                            answers = [answers]
                        if not isinstance(answers, list) or not answers or any(
                            not isinstance(answer, str) or not answer.strip() for answer in answers
                        ):
                            raise SampleRejected("invalid_reference_answers")
                        if sample_id in annotations:
                            used_annotations.add(sample_id)
                        split = stable_split(image_hash, seed, val_fraction, test_fraction)
                        image_splits[image_hash] = split
                        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif",
                                  "BMP": ".bmp", "TIFF": ".tiff"}.get(image_format, ".image")
                        image_path = image_dir / f"{image_hash}{suffix}"
                        if image_path.exists():
                            if sha256_file(image_path) != image_hash:
                                raise ValueError(f"Corrupt existing content-addressed image: {image_path}")
                        else:
                            with image_path.open("xb") as image_file:
                                image_file.write(image_bytes)
                        record = {"format_version": FORMAT_VERSION, "sample_id": sample_id,
                                  "image_hash": image_hash, "image_path": image_path.relative_to(output_dir).as_posix(),
                                  "split": split, "conversations": messages, "task_type": task_type,
                                  "reference_answers": answers, "source_index": source_index,
                                  "source_row": row_index}
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stats["accepted"] += 1
                        stats[f"{split}_records"] += 1
                        stats[f"task_{task_type}"] += 1
                        if official_writer is not None and split == "train":
                            official_indices.append(batch_index)
                    except SampleRejected as exc:
                        stats[f"rejected_{exc.reason}"] += 1
                if official_indices:
                    official_writer.write_table(pa.Table.from_batches([batch]).take(pa.array(official_indices)))
                if max_rows is not None and stats["scanned"] >= max_rows:
                    break
            if max_rows is not None and stats["scanned"] >= max_rows:
                break
        stream.flush()
        os.fsync(stream.fileno())
    if annotations.keys() - used_annotations:
        raise ValueError("Annotations refer to rejected/unscanned/unknown sample IDs")
    if not stats["accepted"]:
        raise ValueError(f"No usable rows: {dict(stats)}")
    os.replace(tmp, manifest)
    if official_target:
        os.replace(official_tmp, official_target)
    summary = {"format_version": FORMAT_VERSION, "status": "complete", "seed": seed,
               "val_fraction": val_fraction, "test_fraction": test_fraction,
               "grouping": "sha256_original_image_bytes", "max_rows": max_rows,
               "sources": sources, "counts": dict(stats), "unique_images": len(image_splits),
               "unique_images_by_split": dict(Counter(image_splits.values())),
               "manifest_sha256": sha256_file(manifest),
               "official_train_parquet": str(official_target) if official_target else None,
               "official_train_parquet_sha256": sha256_file(official_target) if official_target else None,
               "official_train_rows": stats["train_records"] if official_target else None,
               "official_train_policy": "All accepted train-image groups in scanned input; no pilot record cap; original Arrow rows unchanged",
               "open_answer_policy": "No exact-match accuracy for open descriptions."}
    summary_tmp = summary_path.with_suffix(".json.tmp")
    with summary_tmp.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(summary_tmp, summary_path)
    return summary


def iter_manifest(path, split=None):
    """Stream records; pilot consumers can stop before loading the entire manifest."""
    path = Path(path).resolve()
    ids, image_splits = set(), {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("format_version") != FORMAT_VERSION:
                raise ValueError("Unsupported manifest format")
            if row["sample_id"] in ids:
                raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
            ids.add(row["sample_id"])
            if row["split"] not in ("train", "val", "test"):
                raise ValueError("Manifest split must be train, val or test")
            previous = image_splits.setdefault(row["image_hash"], row["split"])
            if previous != row["split"]:
                raise ValueError(f"Image leakage between splits: {row['image_hash']}")
            row["conversations"] = normalize_conversations(row["conversations"])
            row["image_path"] = str((path.parent / row["image_path"]).resolve())
            if split is None or row["split"] == split:
                yield row


def load_manifest(path, split=None) -> list[dict]:
    return list(iter_manifest(path, split=split))


def visual_placeholder(mode="single", image_token=IMAGE_TOKEN) -> str:
    if mode not in ("single", "multi"):
        raise ValueError("mode must be single or multi")
    names = VIEW_NAMES if mode == "multi" else VIEW_NAMES[:1]
    # Text labels also separate runs of markers; five runs must never merge.
    return "\n".join(f"[{name}]\n{image_token * TOKENS_PER_VIEW}" for name in names)


def _encode(tokenizer, text) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def encode_conversation(messages, tokenizer, *, mode="single", max_length=1024,
                        generation=False, reserve_tokens=0, native_single=False) -> dict:
    messages = normalize_conversations(messages)
    marker = _encode(tokenizer, IMAGE_TOKEN)
    if len(marker) != 1 or marker[0] == tokenizer.unk_token_id:
        raise ValueError("Tokenizer must encode <|image_pad|> as one non-UNK token")
    if _encode(tokenizer, tokenizer.eos_token) != [tokenizer.eos_token_id]:
        raise ValueError("Tokenizer EOS must be one token")
    if max_length <= 0 or reserve_tokens < 0:
        raise ValueError("Invalid sequence length or generation reservation")
    input_ids, labels = [], []

    def append(text, supervised=False):
        tokens = _encode(tokenizer, text)
        input_ids.extend(tokens)
        labels.extend(tokens if supervised else [-100] * len(tokens))

    turns = messages[:-1] if generation else messages
    for turn in turns:
        assistant = turn["role"] == "assistant"
        append(f"{tokenizer.bos_token}{turn['role']}\n")
        if assistant:
            append(ASSISTANT_PREFIX)
        placeholder = IMAGE_TOKEN * TOKENS_PER_VIEW if native_single and mode == "single" else visual_placeholder(mode)
        content = turn["content"].replace("<image>", placeholder)
        append(content, assistant)
        append(tokenizer.eos_token, assistant)
        append("\n")
    if generation:
        append(f"{tokenizer.bos_token}assistant\n{ASSISTANT_PREFIX}")
    if len(input_ids) + reserve_tokens > max_length:
        raise SampleRejected("insufficient_sequence_space")
    expected = 5 if mode == "multi" else 1
    runs, index = [], 0
    while index < len(input_ids):
        if input_ids[index] == marker[0]:
            start = index
            while index < len(input_ids) and input_ids[index] == marker[0]:
                index += 1
            runs.append((start, index))
        else:
            index += 1
    if len(runs) != expected or any(end - start != TOKENS_PER_VIEW for start, end in runs):
        raise SampleRejected("invalid_visual_spans")
    if any(labels[index] != -100 for start, end in runs for index in range(start, end)):
        raise ValueError("Visual tokens must not receive answer supervision")
    answer_tokens = sum(label != -100 for label in labels[1:])
    if not generation and not answer_tokens:
        raise SampleRejected("no_supervised_answer_tokens")
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids),
            "answer_tokens": answer_tokens, "visual_spans": runs}


def select_eligible_records(records, tokenizer, *, mode, max_length, max_samples=None,
                            generation=False, reserve_tokens=0, native_single=False):
    """Use MULTI length as a common inclusion rule, including for baseline A."""
    selected, encoded, stats = [], [], Counter()
    for row in records:
        if max_samples is not None and len(selected) >= max_samples:
            break
        stats["examined"] += 1
        try:
            encode_conversation(row["conversations"], tokenizer, mode="multi", max_length=max_length)
            if generation:
                encode_conversation(row["conversations"], tokenizer, mode="multi", max_length=max_length,
                                    generation=True, reserve_tokens=reserve_tokens)
            value = encode_conversation(row["conversations"], tokenizer, mode=mode, max_length=max_length,
                                        generation=generation, reserve_tokens=reserve_tokens, native_single=native_single)
        except SampleRejected as exc:
            stats[f"rejected_{exc.reason}"] += 1
            continue
        selected.append(row)
        for key in ("input_ids", "labels", "attention_mask"):
            value[key] = torch.as_tensor(value[key], dtype=torch.long)
        encoded.append(value)
        stats["accepted"] += 1
    return selected, encoded, dict(stats)


class EvoMindDataset(Dataset):
    def __init__(self, records, encoded, processor=None, *, mode="single", cache=None):
        if len(records) != len(encoded):
            raise ValueError("Records and encodings must be aligned")
        self.records, self.encoded = records, encoded
        self.processor, self.mode, self.cache = processor, mode, cache
        for item in self.encoded:
            for key in ("input_ids", "labels", "attention_mask"):
                item[key] = torch.as_tensor(item[key], dtype=torch.long)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from evomind_v.views import preprocess_views

        row = self.records[index]
        image_bytes = Path(row["image_path"]).read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != row["image_hash"]:
            raise ValueError(f"Image hash mismatch: {row['sample_id']}")
        sample = {k: self.encoded[index][k]
                  for k in ("input_ids", "labels", "attention_mask")}
        sample["record"] = row
        if self.cache is not None:
            features = self.cache.load(self.cache.key_for(image_bytes))
            if features is None:
                raise FileNotFoundError(f"Missing cached features for {row['sample_id']}; build cache first")
            sample["vision_features"] = features
        else:
            with Image.open(io.BytesIO(image_bytes)) as image:
                sample["pixel_values"] = preprocess_views(image, self.processor, mode=self.mode)
        return sample


def collate_samples(samples, pad_token_id):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    result = {"records": [sample["record"] for sample in samples]}
    for name, padding in (("input_ids", pad_token_id), ("labels", -100), ("attention_mask", 0)):
        result[name] = torch.nn.utils.rnn.pad_sequence([sample[name] for sample in samples],
                                                     batch_first=True, padding_value=padding)
    if "vision_features" in samples[0]:
        result["vision_features"] = torch.stack([sample["vision_features"] for sample in samples])
    else:
        result["pixel_values"] = {key: torch.stack([sample["pixel_values"][key] for sample in samples])
                                  for key in samples[0]["pixel_values"]}
    return result


def move_batch(batch, device):
    return {key: ({k: v.to(device) for k, v in value.items()} if isinstance(value, dict) else value.to(device))
            for key, value in batch.items() if key != "records"}
