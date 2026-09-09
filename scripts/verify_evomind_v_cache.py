"""Check real-image B/C feature, output, loss, and projector-gradient parity.

This is a numerical gate before experiments, not a speed benchmark. It loads
the real encoder once, checks batch-size-one five-view samples in eval mode,
and performs no optimizer updates. FP32 cache preparation and online encoding
use the same device; the trainable projector/LLM use the requested AMP dtype.
The online backward graph is released before the cached forward is created.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets  # Match the Windows Arrow/torch DLL initialization order.
import torch
from PIL import Image

from evomind_v.cache import VisionFeatureCache
from evomind_v.data import (EvoMindDataset, collate_samples, iter_manifest,
                           move_batch, select_eligible_records, sha256_file)
from evomind_v.views import preprocess_views
from train_evomind_v import (amp_context, build_runtime, seed_everything,
                             selected_hash, tokenizer_hash)


def tensor_comparison(reference, actual, *, atol, rtol):
    """Report max absolute error and relative L-infinity error.

    The pass condition is elementwise abs(actual-reference) <= atol +
    rtol*abs(reference). Relative L-infinity is max_abs_error divided by
    max(abs(reference)), with a 1e-12 denominator floor, for readability.
    """
    if tuple(reference.shape) != tuple(actual.shape):
        return {"passed": False, "reference_shape": list(reference.shape),
                "actual_shape": list(actual.shape), "reason": "shape_mismatch"}
    left, right = reference.detach().cpu().float(), actual.detach().cpu().float()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return {"passed": False, "shape": list(left.shape), "reason": "nonfinite_tensor"}
    difference = (right - left).abs()
    max_abs = float(difference.max()) if difference.numel() else 0.0
    scale = float(left.abs().max()) if left.numel() else 0.0
    violations = difference > atol + rtol * left.abs()
    return {"passed": not bool(violations.any()), "shape": list(left.shape),
            "max_abs_error": max_abs, "relative_linf_error": max_abs / max(scale, 1e-12),
            "reference_abs_max": scale, "violating_elements": int(violations.sum()),
            "atol": atol, "rtol": rtol}


def select_unique_samples(manifest, tokenizer, *, split, max_length, samples):
    """Require distinct original images, not several questions on one image."""
    records, encoded, seen, counts = [], [], set(), {}
    for row in iter_manifest(manifest, split=split):
        if row["image_hash"] in seen:
            counts["duplicate_image_skipped"] = counts.get("duplicate_image_skipped", 0) + 1
            continue
        chosen, tokens, statistics = select_eligible_records(
            [row], tokenizer, mode="multi", max_length=max_length, max_samples=1)
        for name, number in statistics.items():
            counts[name] = counts.get(name, 0) + number
        if chosen:
            records.extend(chosen)
            encoded.extend(tokens)
            seen.add(row["image_hash"])
            if len(records) == samples:
                break
    return records, encoded, counts


def capture_forward_backward(model, values, args):
    """Keep only CPU detached results; no graph survives this function."""
    model.zero_grad(set_to_none=True)
    with amp_context(args):
        output = model(**values)
    loss = output.loss
    if loss is None:
        raise ValueError("The parity check requires real assistant labels")
    loss.backward()
    gradients = {}
    for name, parameter in model.vision_proj.named_parameters():
        if parameter.grad is None:
            raise ValueError(f"Projector gradient is missing: {name}")
        gradients[name] = parameter.grad.detach().cpu().clone()
    result = {"logits": output.logits.detach().cpu().clone(),
              "loss": loss.detach().cpu().clone(), "gradients": gradients}
    del loss, output
    model.zero_grad(set_to_none=True)
    return result


def verify_parity(model, processor, tokenizer, records, encoded, cache, args):
    """Test a supplied real model; also supports offline CPU dependency fixtures."""
    if model.vision_encoder is None:
        raise ValueError("Verification requires exactly one loaded vision encoder")
    if cache.dtype != "float32" or cache.mode != "multi":
        raise ValueError("The parity gate requires a float32 five-view cache")
    model.requires_grad_(False)
    model.vision_proj.requires_grad_(True)
    model.eval()
    live_dataset = EvoMindDataset(records, encoded, processor, mode="multi")
    observations = {"calls": 0, "capture": False, "features": None}

    def observe_encoder(module, inputs, output):
        observations["calls"] += 1
        if observations["capture"]:
            observations["features"] = output.last_hidden_state.detach().cpu().clone()

    hook = model.vision_encoder.register_forward_hook(observe_encoder)
    results = []
    try:
        for index, row in enumerate(records):
            data = Path(row["image_path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["image_hash"]:
                raise ValueError(f"Image hash mismatch: {row['sample_id']}")
            key = cache.key_for(data)
            features = cache.load(key)
            cache_was_present = features is not None
            roundtrip = None
            if features is None:
                # Process independently from the live dataset path. Reusing a
                # single pixel tensor would not test preprocessing consistency.
                with Image.open(io.BytesIO(data)) as image:
                    pixels = {name: value.unsqueeze(0).to(args.device) for name, value
                              in preprocess_views(image, processor, mode="multi").items()}
                prepared = model.encode_images(pixels)[0].detach().cpu()
                del pixels
                cache.save(key, prepared)
                features = cache.load(key)
                roundtrip = tensor_comparison(prepared, features, atol=0.0, rtol=0.0)
                del prepared
            sample = live_dataset[index]
            values = move_batch(collate_samples([sample], tokenizer.pad_token_id), args.device)
            before_online = observations["calls"]
            observations["capture"] = True
            live = capture_forward_backward(model, values, args)
            observations["capture"] = False
            online_calls = observations["calls"] - before_online
            live_features = observations["features"]
            observations["features"] = None
            values.pop("pixel_values")
            values["vision_features"] = features.unsqueeze(0).to(args.device)
            before_cached = observations["calls"]
            cached = capture_forward_backward(model, values, args)
            cached_calls = observations["calls"] - before_cached
            del values
            comparisons = {
                "features": tensor_comparison(live_features, features, atol=args.feature_atol, rtol=args.feature_rtol),
                "logits": tensor_comparison(live["logits"], cached["logits"], atol=args.atol, rtol=args.rtol),
                "loss": tensor_comparison(live["loss"], cached["loss"], atol=args.atol, rtol=args.rtol),
                "projector_gradients": {
                    name: tensor_comparison(live["gradients"][name], cached["gradients"][name],
                                            atol=args.atol, rtol=args.rtol)
                    for name in live["gradients"]
                },
            }
            if roundtrip is not None:
                comparisons["cache_roundtrip"] = roundtrip
            gradients_nonzero = any(bool(value.abs().sum() > 0) for value in live["gradients"].values())
            encoder_grads_absent = all(parameter.grad is None for parameter in model.vision_encoder.parameters())
            checks = [comparisons[name]["passed"] for name in ("features", "logits", "loss")]
            checks += [value["passed"] for value in comparisons["projector_gradients"].values()]
            if roundtrip is not None:
                checks.append(roundtrip["passed"])
            passed = all(checks) and online_calls == 1 and cached_calls == 0 and gradients_nonzero and encoder_grads_absent
            record = {"sample_id": row["sample_id"], "image_hash": row["image_hash"],
                      "cache_key": key, "cache_was_present": cache_was_present,
                      "online_encoder_calls": online_calls, "cached_encoder_calls": cached_calls,
                      "encoder_gradients_absent": encoder_grads_absent,
                      "projector_gradients_nonzero": gradients_nonzero, "passed": passed,
                      "comparisons": comparisons}
            results.append(record)
            print(json.dumps({"event": "cache_parity_sample", "sample_id": row["sample_id"],
                              "passed": passed, "online_encoder_calls": online_calls,
                              "cached_encoder_calls": cached_calls}), flush=True)
            del live, cached, features, live_features
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
    return results


def atomic_json(payload, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".cache-parity-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", default=str(Path(__file__).resolve().parents[1] / "model"))
    parser.add_argument("--vision-model", required=True)
    parser.add_argument("--init-weights", required=True)
    parser.add_argument("--allow-text-init", action="store_true")
    parser.add_argument("--cache-dir", help="FP32 cache; defaults to OUTPUT parent/features; existing entries are checked, never replaced")
    parser.add_argument("--output-json", "--output", dest="output_json", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--feature-atol", type=float, default=1e-6)
    parser.add_argument("--feature-rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    for name in ("samples", "max_seq_len", "hidden_size", "num_hidden_layers"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("feature_atol", "feature_rtol", "atol", "rtol"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error("Error tolerances must be finite and nonnegative")
    if args.cache_dir is None:
        args.cache_dir = str(Path(args.output_json).resolve().parent / "features")
    for name in ("manifest", "tokenizer", "vision_model", "init_weights", "cache_dir", "output_json"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    args.variant, args.cache_dtype, args.build_cache_only = "B", "float32", False
    return args


def main():
    args = parse_args()
    if Path(args.output_json).exists():
        raise FileExistsError("Use a new --output-json; previous parity evidence is preserved")
    criteria = {"features": {"atol": args.feature_atol, "rtol": args.feature_rtol},
                "logits_loss_projector_gradients": {"atol": args.atol, "rtol": args.rtol},
                "new_cache_roundtrip": {"atol": 0.0, "rtol": 0.0},
                "pass_rule": "All elements satisfy abs(cached-online) <= atol + rtol*abs(online); finite tensors; nonzero projector gradients; one online and zero cached encoder calls"}
    report = {"passed": False, "arguments": vars(args), "criteria_declared_before_run": criteria,
              "batch_size": 1, "image_views": 5, "model_mode": "eval", "cache_dtype": "float32",
              "encoder_compute_dtype": "float32", "optimizer_updates": 0,
              "gradient_scope": "Projector only; LLM parameters frozen, gradient still flows through LLM to projector",
              "torch_version": str(torch.__version__), "results": []}
    print(json.dumps({"event": "cache_parity_plan", "criteria": criteria, "samples": args.samples,
                      "device": args.device, "dtype": args.dtype, "batch_size": 1}), flush=True)
    started = time.perf_counter()
    try:
        seed_everything(args.seed, args.device)
        model, tokenizer, processor, fingerprint, _, initialization = build_runtime(args)
        report.update({"vision_fingerprint": fingerprint, "initialization": initialization,
                       "manifest_sha256": sha256_file(args.manifest),
                       "init_weights_sha256": sha256_file(args.init_weights),
                       "tokenizer_sha256": tokenizer_hash(args.tokenizer),
                       "encoder_class": type(model.vision_encoder).__name__,
                       "model_config": model.config.to_dict(),
                       "backend": {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                                   "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                   "cudnn_benchmark": torch.backends.cudnn.benchmark}})
        if torch.device(args.device).type == "cuda":
            report["device_name"] = torch.cuda.get_device_name(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
        records, encoded, counts = select_unique_samples(
            args.manifest, tokenizer, split=args.split,
            max_length=args.max_seq_len, samples=args.samples)
        if len(records) < args.samples:
            raise ValueError(f"Requested {args.samples} complete eligible samples, found {len(records)}")
        report.update({"selection_counts": counts, "selected_sample_ids_sha256": selected_hash(records)})
        cache = VisionFeatureCache(args.cache_dir, fingerprint, mode="multi", dtype="float32",
                                   hidden_size=model.config.image_hidden_size)
        report["cache_metadata"] = cache.metadata
        report["results"] = verify_parity(model, processor, tokenizer, records, encoded, cache, args)
        report["passed"] = all(row["passed"] for row in report["results"])
        if torch.device(args.device).type == "cuda":
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(report, args.output_json)
    if not report["passed"]:
        raise RuntimeError(f"Cache parity failed its predeclared tolerances; see {args.output_json}")
    print(json.dumps({"event": "cache_parity_passed", "samples": len(report["results"]),
                      "output_json": args.output_json}), flush=True)


if __name__ == "__main__":
    main()
