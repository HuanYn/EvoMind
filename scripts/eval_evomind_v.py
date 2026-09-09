"""Heldout generation, stop/repetition metrics, task-appropriate EM/CER, blind forms.

Open-ended captions never receive an exact-match accuracy score. Human scores
remain blank until a person supplies them; this script does not simulate raters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # Windows DLL import ordering.
import torch

from evomind_v.data import EvoMindDataset, collate_samples, iter_manifest, move_batch, select_eligible_records, sha256_file
from train_evomind_v import (amp_context, build_runtime, load_checkpoint, load_nonvisual_weights,
                             seed_everything, selected_hash, tokenizer_hash)


def normalize_exact(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalize_ocr(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def edit_distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, first in enumerate(left, start=1):
        current = [i]
        for j, second in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (first != second)))
        previous = current
    return previous[-1]


def repeated_ngram_fraction(tokens, n=4):
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def score_prediction(prediction, references, task_type):
    if task_type == "open":
        return {}
    if task_type not in ("closed", "ocr") or not references or any(not isinstance(x, str) or not x.strip() for x in references):
        raise ValueError("EM/CER require an explicit closed/ocr task and nonempty reference answers")
    if task_type == "closed":
        return {"exact_match": int(normalize_exact(prediction) in {normalize_exact(ref) for ref in references})}
    hypothesis = normalize_ocr(prediction)
    targets = [normalize_ocr(ref) for ref in references]
    if any(not target for target in targets):
        raise ValueError("OCR references cannot be empty after normalization")
    alternatives = [(edit_distance(hypothesis, target), len(target)) for target in targets]
    distance, characters = min(alternatives, key=lambda pair: (pair[0] / pair[1], pair[0]))
    return {"ocr_exact_match": int(hypothesis in targets), "character_edits": distance,
            "reference_characters": characters, "character_error_rate": distance / characters}


def summarize_predictions(rows):
    if not rows:
        raise ValueError("No predictions to summarize")
    closed = [row for row in rows if row["task_type"] == "closed"]
    ocr = [row for row in rows if row["task_type"] == "ocr"]
    open_rows = [row for row in rows if row["task_type"] == "open"]
    return {"samples": len(rows), "unique_images": len({row["image_hash"] for row in rows}),
            "eos_rate": sum(row["ended_with_eos"] for row in rows) / len(rows),
            "max_tokens_reached_rate": sum(row["stop_reason"] == "max_new_tokens" for row in rows) / len(rows),
            "mean_repeated_4gram_fraction": sum(row["repeated_4gram_fraction"] for row in rows) / len(rows),
            "closed_samples": len(closed), "closed_answer_exact_match":
            sum(row["exact_match"] for row in closed) / len(closed) if closed else None,
            "ocr_samples": len(ocr), "ocr_exact_match":
            sum(row["ocr_exact_match"] for row in ocr) / len(ocr) if ocr else None,
            "ocr_character_error_rate": sum(row["character_edits"] for row in ocr) /
            sum(row["reference_characters"] for row in ocr) if ocr else None,
            "open_samples": len(open_rows), "human_evaluation_status": "pending_manual_annotation",
            "human_raters": 0, "open_answer_accuracy": None}


def export_blind_forms(rows, output_dir, checkpoint_hash, seed):
    order = list(rows)
    random.Random(seed).shuffle(order)
    keys = []
    fields = ["response_id", "sample_id", "image_path", "question", "conversation_context", "prediction",
              "correctness_1_to_5", "grounding_1_to_5", "hallucination_yes_no", "notes"]
    with (Path(output_dir) / "blind_annotation.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in order:
            response_id = hashlib.sha256(f"{seed}:{checkpoint_hash}:{row['sample_id']}".encode()).hexdigest()[:16]
            writer.writerow({"response_id": response_id, "sample_id": row["sample_id"],
                             "image_path": row["image_path"], "question": row["question"],
                             "conversation_context": json.dumps(row.get("conversation_context", []), ensure_ascii=False),
                             "prediction": row["prediction"]})
            keys.append({"response_id": response_id, "sample_id": row["sample_id"], "checkpoint_sha256": checkpoint_hash})
    with (Path(output_dir) / "blind_key.json").open("x", encoding="utf-8") as stream:
        json.dump(keys, stream, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument("--checkpoint", help="EvoMind-V training checkpoint with saved provenance")
    artifact.add_argument("--official-weights", help="Official MiniMind-V plain state_dict; reference evaluation only")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--official-examples", action="store_true", help="Use six pinned eval_vlm.py examples; no blind forms or custom heldout benchmark")
    parser.add_argument("--manifest", help="Must have the same SHA256 as the training manifest")
    parser.add_argument("--vision-model", help="Optional relocated local encoder directory; fingerprint must match")
    parser.add_argument("--tokenizer", help="Optional relocated tokenizer directory; content hash must match")
    parser.add_argument("--cache-dir", help="Optional relocated feature cache for C")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Final report uses untouched test; val is for development only")
    parser.add_argument("--max-samples", "--eval", dest="max_samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=768, help="Official plain-weight architecture only")
    parser.add_argument("--num-hidden-layers", type=int, default=8, help="Official plain-weight architecture only")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Official reference common eligibility budget")
    args = parser.parse_args()
    if args.max_samples <= 0 or args.max_new_tokens <= 0:
        parser.error("Evaluation sample and token limits must be positive")
    if args.official_weights and (not args.manifest or not args.vision_model):
        parser.error("--official-weights requires --manifest and --vision-model")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed, args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Evaluation requires an empty output directory")
    if args.official_weights:
        tokenizer_path = args.tokenizer or str(Path(__file__).resolve().parents[1] / "model")
        weights = torch.load(args.official_weights, map_location="cpu", weights_only=True)
        if isinstance(weights, dict) and "model" in weights:
            weights = weights["model"]
        checkpoint = {"model": weights, "global_step": None,
                      "config": {"arguments": {"variant": "A", "manifest": str(Path(args.manifest).resolve()),
                          "tokenizer": str(Path(tokenizer_path).resolve()), "vision_model": str(Path(args.vision_model).resolve()),
                          "cache_dir": None, "cache_dtype": "float32", "hidden_size": args.hidden_size,
                          "num_hidden_layers": args.num_hidden_layers, "max_seq_len": args.max_seq_len,
                          "seed": args.seed, "dtype": args.dtype or "bfloat16", "allow_text_init": False},
                          "manifest_sha256": sha256_file(args.manifest), "tokenizer_sha256": tokenizer_hash(tokenizer_path),
                          "vision_fingerprint": None}}
    else:
        checkpoint = load_checkpoint(args.checkpoint)
    saved = checkpoint["config"]
    runtime_args = argparse.Namespace(**saved["arguments"])
    runtime_args.device = args.device
    runtime_args.build_cache_only = False
    if args.dtype:
        runtime_args.dtype = args.dtype
    for field in ("manifest", "vision_model", "tokenizer", "cache_dir"):
        if getattr(args, field):
            setattr(runtime_args, field, str(Path(getattr(args, field)).resolve()))
    if sha256_file(runtime_args.manifest) != saved["manifest_sha256"]:
        raise ValueError("Heldout manifest differs from the training manifest")
    if tokenizer_hash(runtime_args.tokenizer) != saved["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the training tokenizer")
    report_runtime_variant = runtime_args.variant
    if args.official_examples and runtime_args.variant == "C":
        runtime_args.variant = "B"  # New images use frozen encoder online, never a missing training cache.
    model, tokenizer, processor, fingerprint, cache, _ = build_runtime(runtime_args, initialize=False)
    runtime_args.variant = report_runtime_variant
    if saved["vision_fingerprint"] is not None and fingerprint != saved["vision_fingerprint"]:
        raise ValueError("Image preprocessing/encoder fingerprint differs from training")
    load_nonvisual_weights(model, checkpoint["model"])
    model.eval()
    mode = "single" if runtime_args.variant == "A" else "multi"
    report_variant = "official_reference" if args.official_weights else runtime_args.variant
    official_source = None
    if args.official_examples:
        from evomind_v.official_eval import examples
        candidates, official_source = examples(Path(__file__).resolve().parents[1])
        runtime_args.max_seq_len = max(1024, runtime_args.max_seq_len)
    else:
        candidates = iter_manifest(runtime_args.manifest, split=args.split)
    records, encoded, filter_stats = select_eligible_records(
        candidates, tokenizer, mode=mode,
        max_length=runtime_args.max_seq_len, max_samples=args.max_samples,
        generation=True, reserve_tokens=args.max_new_tokens, native_single=bool(args.official_weights))
    if args.official_examples and len(records) != 6:
        raise ValueError("All six upstream images must fit; do not silently drop a case")
    if not records:
        raise ValueError(f"No heldout samples fit the common five-view generation budget: {filter_stats}")
    dataset = EvoMindDataset(records, encoded, processor, mode=mode, cache=cache)
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    checkpoint_hash = sha256_file(args.official_weights or args.checkpoint)
    predictions = []
    started = time.perf_counter()
    with (output_dir / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for index in range(len(dataset)):
            sample = dataset[index]
            row = sample["record"]
            values = move_batch(collate_samples([sample], tokenizer.pad_token_id), args.device)
            values.pop("labels")
            prompt_length = values["input_ids"].shape[1]
            if torch.device(args.device).type == "cuda":
                torch.cuda.synchronize(args.device)
            sample_start = time.perf_counter()
            with torch.inference_mode(), amp_context(runtime_args):
                generated = model.generate(**values, max_new_tokens=args.max_new_tokens, do_sample=args.official_examples,
                                           temperature=.7 if args.official_examples else 1.0, top_p=.85 if args.official_examples else 1.0, top_k=50 if args.official_examples else 0, repetition_penalty=1.0,
                                           eos_token_id=tokenizer.eos_token_id, use_cache=True, logits_to_keep=1)
            if torch.device(args.device).type == "cuda":
                torch.cuda.synchronize(args.device)
            latency = time.perf_counter() - sample_start
            tokens = generated[0, prompt_length:].tolist()
            ended = tokenizer.eos_token_id in tokens
            content_tokens = tokens[:tokens.index(tokenizer.eos_token_id)] if ended else tokens
            prediction = tokenizer.decode(content_tokens, skip_special_tokens=True)
            result = {"sample_id": row["sample_id"], "image_hash": row["image_hash"], "image_path": row["image_path"],
                      "task_type": row.get("task_type", "open"), "question": row["conversations"][-2]["content"],
                      "conversation_context": row["conversations"][:-1],
                      "reference_answers": row["reference_answers"], "prediction": prediction,
                      "generated_token_ids": tokens, "generated_tokens": len(tokens), "prompt_tokens": prompt_length,
                      "ended_with_eos": ended, "stop_reason": "eos" if ended else "max_new_tokens",
                      "repeated_4gram_fraction": repeated_ngram_fraction(content_tokens), "generation_seconds": latency,
                      "variant": report_variant, "seed": args.seed, "training_seed": None if args.official_weights else runtime_args.seed,
                      "global_step": checkpoint["global_step"]}
            result.update(score_prediction(prediction, row["reference_answers"], result["task_type"]))
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            predictions.append(result)
            if (index + 1) % 10 == 0:
                print(json.dumps({"event": "eval_progress", "completed": index + 1, "total": len(dataset)}), flush=True)
    summary = {**summarize_predictions(predictions), "variant": report_variant, "seed": args.seed,
               "training_seed": None if args.official_weights else runtime_args.seed, "split": "official_examples" if args.official_examples else args.split,
               "official_examples": official_source,
               "weight_format": "official_plain_state_dict" if args.official_weights else "evomind_checkpoint",
               "training_provenance": "Plain official weights cannot verify their training split; runner must retain its train-only manifest provenance. Different budgets are not a controlled A/B/C gain."
               if args.official_weights else "Checkpoint manifest, tokenizer and encoder fingerprints verified",
               "checkpoint_sha256": checkpoint_hash, "global_step": checkpoint["global_step"],
               "manifest_sha256": saved["manifest_sha256"], "heldout_sample_ids_sha256": selected_hash(records),
               "filter_counts": filter_stats, "image_mode": mode, "vision_fingerprint": fingerprint,
               "max_new_tokens": args.max_new_tokens, "inference_dtype": runtime_args.dtype,
               "max_seq_len": runtime_args.max_seq_len,
               "prompt_format": "official_native_single" if args.official_weights else "evomind_labelled_views",
               "decoding": "sampled; temperature=.7; top_k=50; top_p=.85; repetition_penalty=1; fixed seed" if args.official_examples else "greedy; temperature=1; top_k=0; top_p=1; repetition_penalty=1",
               "closed_normalization": "Unicode NFKC + casefold + whitespace collapse",
               "ocr_normalization": "Unicode NFKC + remove whitespace; preserve case; corpus CER over best accepted references",
               "repetition_definition": "1 - unique generated token 4-grams / all generated token 4-grams; exclude EOS",
               "wall_seconds": time.perf_counter() - started,
               "mean_generation_seconds": sum(row["generation_seconds"] for row in predictions) / len(predictions),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if torch.device(args.device).type == "cuda" else None,
               "human_annotation_note": "Blank randomized forms only. Keep blind_key.json and model directories from raters; merge/shuffle candidates across runs before comparative human evaluation."}
    if args.official_examples:
        summary.update(human_evaluation_status="cancelled_by_user",
                       human_annotation_note="No blind review requested. Upstream examples are not a representative quality benchmark.",
                       inference_cache="online frozen encoder for all variants, including C")
    else:
        export_blind_forms(predictions, output_dir, checkpoint_hash, args.seed)
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
