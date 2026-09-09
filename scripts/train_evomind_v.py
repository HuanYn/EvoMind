"""Reproducible A (single), B (five-view), C (five-view cache) SFT.

Loss means and accumulated gradients are weighted by the number of non-ignored
shifted assistant target tokens, including genuine assistant EOS tokens. Saving
occurs only after optimizer updates, never halfway through an accumulation window.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets  # Windows: initialize Arrow/datasets DLL dependencies before torch.
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, SiglipImageProcessor

from evomind_v.cache import VisionFeatureCache, fingerprint_vision
from evomind_v.data import (EvoMindDataset, collate_samples, iter_manifest, move_batch,
                           select_eligible_records, sha256_file)
from evomind_v.model import EvoMindVLM
from evomind_v.views import preprocess_views
from model.model_vlm import VLMConfig

CHECKPOINT_VERSION = 1


def seed_everything(seed, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng(device="cpu"):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(),
            "cuda": {"device": str(device), "state": torch.cuda.get_rng_state(device)}
            if torch.device(device).type == "cuda" else None}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("This checkpoint requires a CUDA RNG state")
        torch.cuda.set_rng_state(state["cuda"]["state"].cpu(), device=state["cuda"]["device"])


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def atomic_save(payload, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Not a compatible EvoMind-V training checkpoint")
    for field in ("model", "optimizer", "scaler", "config", "rng", "epoch", "next_batch", "global_step"):
        if field not in checkpoint:
            raise ValueError(f"Checkpoint missing {field}")
    return checkpoint


def load_nonvisual_weights(model, weights, *, allow_missing_projector=False):
    if not isinstance(weights, dict):
        raise ValueError("Weights must be a tensor state dict")
    cleaned = {key: value for key, value in weights.items() if not key.startswith("vision_encoder.")}
    if not cleaned or any(not isinstance(value, torch.Tensor) for value in cleaned.values()):
        raise ValueError("Checkpoint contains non-tensor model weights")
    result = model.load_state_dict(cleaned, strict=False)
    allowed = ("vision_encoder.", "vision_proj.") if allow_missing_projector else ("vision_encoder.",)
    missing = [name for name in result.missing_keys if not name.startswith(allowed)]
    if missing or result.unexpected_keys:
        raise ValueError(f"Incompatible base/checkpoint: missing={missing}, unexpected={result.unexpected_keys}")
    return {"missing_projector": [name for name in result.missing_keys if name.startswith("vision_proj.")],
            "loaded_tensors": len(cleaned)}


def freeze_for_sft(model):
    model.requires_grad_(False)
    model.vision_proj.requires_grad_(True)
    model.model.layers[0].requires_grad_(True)
    model.model.layers[-1].requires_grad_(True)
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def reset_projector_deterministically(projector, seed):
    """Encoder construction must not change the text-init projector across A/B/C."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in projector.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


def module_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_runtime(args, *, initialize=True):
    """Load only local models. C loads preprocessing config but no encoder module."""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    marker = tokenizer.encode("<|image_pad|>", add_special_tokens=False)
    if len(marker) != 1 or marker[0] == tokenizer.unk_token_id:
        raise ValueError("Image marker is not a single tokenizer token")
    processor = SiglipImageProcessor.from_pretrained(args.vision_model, local_files_only=True)
    fingerprint = fingerprint_vision(args.vision_model, processor)
    config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                       max_seq_len=args.max_seq_len, use_moe=False, image_ids=marker,
                       vocab_size=len(tokenizer), bos_token_id=tokenizer.bos_token_id,
                       eos_token_id=tokenizer.eos_token_id)
    cached = args.variant == "C"
    model = EvoMindVLM(config, vision_model_path=args.vision_model, processor=processor,
                       load_vision_encoder=not cached or getattr(args, "build_cache_only", False))
    if (not cached or getattr(args, "build_cache_only", False)) and model.vision_encoder is None:
        raise ValueError(f"Could not load local vision encoder: {args.vision_model}")
    initialization = None
    if initialize:
        weights = torch.load(args.init_weights, map_location="cpu", weights_only=True)
        if isinstance(weights, dict) and "model" in weights:
            weights = weights["model"]
        initialization = load_nonvisual_weights(model, weights, allow_missing_projector=args.allow_text_init)
        if initialization["missing_projector"]:
            if set(initialization["missing_projector"]) != {"vision_proj." + name for name in model.vision_proj.state_dict()}:
                raise ValueError("Text initialization must omit the whole projector, not partially initialize it")
            reset_projector_deterministically(model.vision_proj, args.seed)
        initialization["projector_sha256"] = module_hash(model.vision_proj)
    freeze_for_sft(model)
    cache = None
    if cached:
        if not args.cache_dir:
            raise ValueError("Variant C requires --cache-dir containing precomputed five-view features")
        cache = VisionFeatureCache(args.cache_dir, fingerprint, mode="multi", dtype=args.cache_dtype)
    model.to(args.device)
    return model, tokenizer, processor, fingerprint, cache, initialization


def amp_context(args):
    device_type = torch.device(args.device).type
    if args.dtype == "float32":
        return nullcontext()
    if device_type == "cpu" and args.dtype == "float16":
        raise ValueError("CPU float16 is unsupported; use float32 or bfloat16")
    return torch.autocast(device_type=device_type, dtype=getattr(torch, args.dtype))


def append_metric(path, payload):
    line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()
    print(line, flush=True)


def selected_hash(records):
    return hashlib.sha256("\n".join(row["sample_id"] for row in records).encode()).hexdigest()


def tokenizer_hash(path):
    directory = Path(path)
    files = sorted(file for file in directory.iterdir() if file.is_file() and
                   ("token" in file.name or file.name in ("vocab.json", "merges.txt", "chat_template.jinja")))
    if not files:
        raise ValueError("Tokenizer directory has no identifiable tokenizer files")
    return hashlib.sha256(json.dumps([(file.name, sha256_file(file)) for file in files]).encode()).hexdigest()


def experiment_config(args, model, fingerprint, train_records, val_records, planned_steps):
    arguments = vars(args).copy()
    # These control stopping/logging/storage, and may change when continuing.
    for key in ("resume", "output_dir", "max_steps", "save_every", "eval_every", "build_cache_only", "cache_budget_gb",
                "max_test_samples", "cache_eval_samples", "cache_eval_max_new_tokens"):
        arguments.pop(key, None)
    source_root = Path(__file__).resolve().parents[1]
    code_files = [Path(__file__), *(source_root / "evomind_v" / name for name in ("data.py", "model.py", "views.py", "cache.py"))]
    return {"arguments": arguments, "model_config": model.config.to_dict(),
            "manifest_sha256": sha256_file(args.manifest), "init_weights_sha256": sha256_file(args.init_weights),
            "tokenizer_sha256": tokenizer_hash(args.tokenizer), "vision_fingerprint": fingerprint,
            "train_sample_ids_sha256": selected_hash(train_records), "val_sample_ids_sha256": selected_hash(val_records),
            "train_records": len(train_records), "val_records": len(val_records), "planned_optimizer_steps": planned_steps,
            "code_sha256": {str(path.relative_to(source_root)): sha256_file(path) for path in code_files},
            "torch_version": str(torch.__version__),
            "loss_definition": "sum assistant next-token NLL / count of non-ignored shifted assistant target tokens; includes true EOS"}


def evaluate_nll(model, dataset, args, tokenizer):
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=partial(collate_samples, pad_token_id=tokenizer.pad_token_id),
                        generator=torch.Generator().manual_seed(args.seed + 900001))
    with torch.no_grad():
        for batch in loader:
            values = move_batch(batch, args.device)
            count = int((values["labels"][:, 1:] != -100).sum().item())
            with amp_context(args):
                output = model(**values)
            total_nll += float(output.loss.item()) * count
            total_tokens += count
    model.train(was_training)
    if not total_tokens:
        raise ValueError("Heldout evaluation has no answer tokens")
    return {"answer_nll": total_nll / total_tokens, "answer_nll_sum": total_nll,
            "answer_tokens": total_tokens, "samples": len(dataset)}


def save_training_checkpoint(model, optimizer, scaler, config, *, epoch, next_batch,
                             global_step, totals, output_dir, save_best=False):
    payload = {"checkpoint_version": CHECKPOINT_VERSION,
               "model": cpu_tree({key: value for key, value in model.state_dict().items()
                                   if not key.startswith("vision_encoder.")}),
               "optimizer": cpu_tree(optimizer.state_dict()), "scaler": scaler.state_dict(),
               "config": config, "rng": capture_rng(next(model.parameters()).device), "epoch": epoch, "next_batch": next_batch,
               "global_step": global_step, "totals": totals}
    destination = Path(output_dir) / f"step_{global_step:06d}.pt"
    # Keep the last two step snapshots plus best validation, never select on test.
    atomic_save(payload, destination)
    atomic_save(payload, Path(output_dir) / "last.pt")
    if save_best:
        atomic_save(payload, Path(output_dir) / "best_val.pt")
    import re
    owned_root = Path(output_dir).resolve()
    snapshots = sorted(path for path in owned_root.glob("step_*.pt")
                       if re.fullmatch(r"step_\d{6,}\.pt", path.name) and not path.is_symlink())
    for old in snapshots[:-2]:
        if old.resolve().parent != owned_root:
            raise ValueError("Checkpoint retention target escaped output directory")
        old.unlink()
    return destination


def build_selected_cache(model, processor, cache, records, args):
    unique = {record["image_hash"]: record for record in records}
    bytes_per_image = math.prod(cache.shape) * (4 if cache.dtype == "float32" else 2)
    estimate = bytes_per_image * len(unique)
    print(json.dumps({"event": "cache_plan", "unique_images": len(unique), "raw_tensor_bytes": estimate,
                      "raw_tensor_gib": estimate / 2**30, "budget_gib": args.cache_budget_gb,
                      "note": "Serialization/filesystem overhead is additional; encoder compute is FP32."}), flush=True)
    if estimate > args.cache_budget_gb * 2**30:
        raise ValueError("Estimated cache exceeds --cache-budget-gb; select fewer records or explicitly raise budget")
    cache.root.mkdir(parents=True, exist_ok=True)
    import shutil
    if shutil.disk_usage(cache.root).free < estimate * 1.05:
        raise OSError("Insufficient free space for the conservative cache size estimate")
    started, existing, created = time.perf_counter(), 0, 0
    model.eval()
    for row in unique.values():
        data = Path(row["image_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["image_hash"]:
            raise ValueError(f"Image hash mismatch: {row['sample_id']}")
        key = cache.key_for(data)
        if cache.load(key) is not None:
            existing += 1
            continue
        with Image.open(io.BytesIO(data)) as image:
            pixels = {name: value.unsqueeze(0).to(args.device)
                      for name, value in preprocess_views(image, processor, mode="multi").items()}
        features = model.encode_images(pixels)[0]
        cache.save(key, features.detach())
        created += 1
        if created % 100 == 0:
            print(json.dumps({"event": "cache_progress", "created": created, "existing": existing,
                              "unique_images": len(unique), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    summary = {"event": "cache_complete", "created": created, "existing": existing,
               "unique_images": len(unique), "raw_tensor_bytes": estimate,
               "elapsed_seconds": time.perf_counter() - started, "metadata": cache.metadata,
               "manifest_sha256": sha256_file(args.manifest), "selected_sample_ids_sha256": selected_hash(records)}
    with (Path(args.output_dir) / "cache_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", default=str(Path(__file__).resolve().parents[1] / "model"))
    parser.add_argument("--vision-model", required=True)
    parser.add_argument("--init-weights", required=True)
    parser.add_argument("--allow-text-init", action="store_true", help="Explicitly allow a text-only base with a random projector")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--build-cache-only", action="store_true", help="Build C features for the selected train/val records, without training")
    parser.add_argument("--cache-budget-gb", type=float, default=64.0, help="Maximum estimated tensor GiB for explicit cache preparation")
    parser.add_argument("--resume", help="An EvoMind-V step_N.pt or last.pt checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, help="Stop after this total number of optimizer updates; does not alter LR horizon")
    parser.add_argument("--max-train-samples", "--max-train-records", dest="max_train_samples", type=int)
    parser.add_argument("--max-val-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=500, help="Cache preparation only; never used by training/validation")
    parser.add_argument("--cache-eval-samples", type=int, default=500, help="Additional development-val generation samples to cache")
    parser.add_argument("--cache-eval-max-new-tokens", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    for field in ("epochs", "batch_size", "grad_accum", "max_seq_len", "save_every", "eval_every"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    for field in ("max_steps", "max_train_samples", "max_val_samples", "max_test_samples", "cache_eval_samples", "cache_eval_max_new_tokens"):
        if getattr(args, field) is not None and getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.grad_clip <= 0 or args.weight_decay < 0 or args.num_workers < 0:
        parser.error("Invalid optimizer or worker setting")
    if args.cache_budget_gb <= 0 or (args.build_cache_only and args.variant != "C"):
        parser.error("Cache budget must be positive; --build-cache-only requires --variant C")
    if args.build_cache_only and args.resume:
        parser.error("Cache preparation does not use training --resume")
    for field in ("manifest", "tokenizer", "vision_model", "init_weights", "output_dir", "cache_dir", "resume"):
        if getattr(args, field):
            setattr(args, field, str(Path(getattr(args, field)).resolve()))
    return args


def main():
    args = parse_args()
    seed_everything(args.seed, args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output_dir.iterdir()):
        raise FileExistsError("Use an empty --output-dir for a new run, or pass --resume")
    resume = load_checkpoint(args.resume) if args.resume else None
    model, tokenizer, processor, fingerprint, cache, initialization = build_runtime(
        args, initialize=resume is None and not args.build_cache_only)
    mode = "single" if args.variant == "A" else "multi"
    print(json.dumps({"event": "data_memory_plan", "manifest_bytes": Path(args.manifest).stat().st_size,
                      "max_train_records": args.max_train_samples, "max_val_records": args.max_val_samples,
                      "max_encoded_tensor_bytes": (args.max_train_samples + args.max_val_samples) * args.max_seq_len * 24
                      if args.max_train_samples is not None else None,
                      "note": "Manifest streams; selected records and encoded tensors stay in RAM. Full uncapped runs require RAM proportional to accepted token count."}), flush=True)
    train_records, train_encoded, train_stats = select_eligible_records(
        iter_manifest(args.manifest, split="train"), tokenizer, mode=mode,
        max_length=args.max_seq_len, max_samples=args.max_train_samples)
    val_records, val_encoded, val_stats = select_eligible_records(
        iter_manifest(args.manifest, split="val"), tokenizer, mode=mode,
        max_length=args.max_seq_len, max_samples=args.max_val_samples)
    if not train_records or not val_records:
        raise ValueError(f"Need nonempty train and heldout splits after length checks: train={train_stats}, val={val_stats}")
    if args.build_cache_only:
        val_eval_records, _, val_eval_stats = select_eligible_records(
            iter_manifest(args.manifest, split="val"), tokenizer, mode="multi", max_length=args.max_seq_len,
            max_samples=args.cache_eval_samples, generation=True, reserve_tokens=args.cache_eval_max_new_tokens)
        test_records, _, test_stats = select_eligible_records(
            iter_manifest(args.manifest, split="test"), tokenizer, mode="multi", max_length=args.max_seq_len,
            max_samples=args.max_test_samples, generation=True, reserve_tokens=args.cache_eval_max_new_tokens)
        print(json.dumps({"event": "cache_split_counts", "train": train_stats, "val_nll": val_stats,
                          "val_generation": val_eval_stats, "test_generation": test_stats}), flush=True)
        build_selected_cache(model, processor, cache, train_records + val_records + val_eval_records + test_records, args)
        return
    train_dataset = EvoMindDataset(train_records, train_encoded, processor, mode=mode, cache=cache)
    val_dataset = EvoMindDataset(val_records, val_encoded, processor, mode=mode, cache=cache)
    batches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    planned_steps = args.epochs * math.ceil(batches_per_epoch / args.grad_accum)
    config = experiment_config(args, model, fingerprint, train_records, val_records, planned_steps)
    config["initial_projector_sha256"] = resume["config"]["initial_projector_sha256"] if resume else module_hash(model.vision_proj)
    if resume and resume["config"] != config:
        changed = [key for key in config if resume["config"].get(key) != config[key]]
        raise ValueError(f"Unsafe resume: experiment configuration changed: {changed}")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.device(args.device).type == "cuda" and args.dtype == "float16")
    epoch_start, skip_batches, global_step = 0, 0, 0
    totals = {"answer_nll_sum": 0.0, "answer_tokens": 0, "samples": 0, "optimizer_steps": 0,
              "best_validation_nll": None}
    if resume:
        load_nonvisual_weights(model, resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        epoch_start, skip_batches, global_step = resume["epoch"], resume["next_batch"], resume["global_step"]
        totals = resume["totals"]
        restore_rng(resume["rng"])
    else:
        # Model/encoder construction consumes different RNG streams in A/B vs C.
        seed_everything(args.seed, args.device)
        with (output_dir / "run_config.json").open("x", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
    metrics_path = output_dir / "metrics.jsonl"
    append_metric(metrics_path, {"event": "start" if not resume else "resume", "variant": args.variant,
                               "global_step": global_step, "train_filter": train_stats, "val_filter": val_stats,
                               "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                               "initialization": initialization, "planned_steps": planned_steps,
                               "encoder_loaded": model.vision_encoder is not None})
    if args.max_steps is not None and global_step >= args.max_steps:
        print("Checkpoint already reached --max-steps; no new updates requested", flush=True)
        return
    if epoch_start >= args.epochs:
        print("Checkpoint already completed all configured epochs", flush=True)
        return
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    stop = False
    for epoch in range(epoch_start, args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        permutation = torch.randperm(len(train_dataset), generator=generator).tolist()
        batches = [permutation[index:index + args.batch_size] for index in range(0, len(permutation), args.batch_size)]
        skip = skip_batches if epoch == epoch_start else 0
        loader = DataLoader(train_dataset, batch_sampler=batches[skip:], num_workers=args.num_workers,
                            pin_memory=torch.device(args.device).type == "cuda",
                            collate_fn=partial(collate_samples, pad_token_id=tokenizer.pad_token_id),
                            generator=torch.Generator().manual_seed(args.seed + epoch + 800001))
        window_nll, window_tokens, window_samples, window_batches = 0.0, 0, 0, 0
        for batch_index, batch in enumerate(loader, start=skip):
            values = move_batch(batch, args.device)
            token_count = int((values["labels"][:, 1:] != -100).sum().item())
            if token_count == 0:
                raise ValueError("Batch has no assistant target tokens")
            with amp_context(args):
                output = model(**values)
                token_sum_loss = output.loss * token_count
            if not torch.isfinite(token_sum_loss):
                raise FloatingPointError("Non-finite answer loss; last completed checkpoint remains available")
            scaler.scale(token_sum_loss).backward()
            window_nll += float(output.loss.detach().item()) * token_count
            window_tokens += token_count
            window_samples += len(batch["records"])
            window_batches += 1
            del output, token_sum_loss, values
            at_boundary = window_batches == args.grad_accum or batch_index + 1 == batches_per_epoch
            if not at_boundary:
                continue
            scaler.unscale_(optimizer)
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(window_tokens)
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
            lr = args.learning_rate * (0.1 + 0.45 * (1 + math.cos(math.pi * global_step / max(planned_steps, 1))))
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            totals["answer_nll_sum"] += window_nll
            totals["answer_tokens"] += window_tokens
            totals["samples"] += window_samples
            totals["optimizer_steps"] += 1
            elapsed = time.perf_counter() - started
            append_metric(metrics_path, {"event": "train", "variant": args.variant, "epoch": epoch,
                                       "next_batch": batch_index + 1, "global_step": global_step,
                                       "answer_nll": window_nll / window_tokens, "answer_nll_sum": window_nll,
                                       "answer_tokens": window_tokens, "samples": window_samples,
                                       "micro_batches": window_batches, "learning_rate": lr,
                                       "gradient_norm": float(gradient_norm), "elapsed_seconds": elapsed,
                                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device)
                                       if torch.device(args.device).type == "cuda" else None})
            window_nll, window_tokens, window_samples, window_batches = 0.0, 0, 0, 0
            last_batch = batch_index + 1 == batches_per_epoch
            stop = args.max_steps is not None and global_step >= args.max_steps
            improved_validation = False
            if global_step % args.eval_every == 0 or stop or last_batch:
                measured = evaluate_nll(model, val_dataset, args, tokenizer)
                if totals["best_validation_nll"] is None or measured["answer_nll"] < totals["best_validation_nll"]:
                    totals["best_validation_nll"] = measured["answer_nll"]
                    improved_validation = True
                append_metric(metrics_path, {"event": "validation", "variant": args.variant,
                                           "global_step": global_step, **measured})
            if global_step % args.save_every == 0 or stop or last_batch or improved_validation:
                next_epoch, next_batch = (epoch + 1, 0) if last_batch else (epoch, batch_index + 1)
                checkpoint_path = save_training_checkpoint(
                    model, optimizer, scaler, config, epoch=next_epoch, next_batch=next_batch,
                    global_step=global_step, totals=totals, output_dir=output_dir, save_best=improved_validation)
                append_metric(metrics_path, {"event": "checkpoint", "global_step": global_step, "path": str(checkpoint_path)})
            if stop:
                break
        if stop:
            break
    append_metric(metrics_path, {"event": "complete" if not stop or global_step >= planned_steps else "probe_stopped",
                               "global_step": global_step, "totals": totals,
                               "mean_answer_nll": totals["answer_nll_sum"] / totals["answer_tokens"],
                               "wall_seconds_this_process": time.perf_counter() - started})


if __name__ == "__main__":
    main()
