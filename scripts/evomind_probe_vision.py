"""Probe full official MiniMind-V training memory; never save trained weights.

Run one explicit configuration per invocation after text SFT has completed:
  python scripts/evomind_probe_vision.py --batch 4
  python scripts/evomind_probe_vision.py --batch 1 --accumulation-steps 4

An OOM is written to JSON and exits 3; there is no automatic fallback. Inputs
are a bounded real-data batch processed by the official VLMDataset. Reported
losses are probe diagnostics, NOT model quality or validation measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "vision"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command_output(argv):
    result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=20)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {argv}: {result.stderr.strip()}")
    return result.stdout.strip()


def select_single_image_rows(parquet_path, batch_size, limit=512):
    """Read at most limit real rows; no silent substitution of synthetic images."""
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(parquet_path)
    rows, indices, examined = [], [], 0
    for batch in parquet.iter_batches(batch_size=32, columns=["conversations", "image_bytes"]):
        for row in batch.to_pylist():
            index = examined
            examined += 1
            messages = row["conversations"]
            messages = json.loads(messages) if isinstance(messages, str) else messages
            pictures = row["image_bytes"]
            pictures = pictures if isinstance(pictures, list) else [pictures]
            markers = sum(turn.get("content", "").count("<image>") for turn in messages)
            if len(pictures) == 1 and pictures[0] and markers == 1 and any(
                    turn.get("role") == "assistant" and turn.get("content") for turn in messages):
                rows.append(row)
                indices.append(index)
                if len(rows) == batch_size:
                    return rows, indices, parquet.metadata.num_rows
            if examined >= limit:
                raise RuntimeError(f"Fewer than {batch_size} valid single-image examples in the first {limit} rows.")
    raise RuntimeError(f"Dataset has fewer than {batch_size} valid single-image examples.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=[1, 4], default=4)
    parser.add_argument("--seq", type=int, choices=[768], default=768)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3, help="Optimizer updates, including the first warmup update; minimum 2")
    parser.add_argument("--text-weight", type=Path, default=ROOT / "out" / "full_sft_768.pth")
    parser.add_argument("--vision-model", type=Path, default=VISION / "model" / "siglip2-base-p32-256-ve")
    parser.add_argument("--data-path", type=Path, default=VISION / "dataset" / "sft_i2t.parquet")
    parser.add_argument("--output", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    if args.steps < 2 or args.accumulation_steps < 1:
        raise SystemExit("--steps must be at least 2 and --accumulation-steps must be positive")
    args.text_weight, args.vision_model, args.data_path = (
        path.resolve() for path in (args.text_weight, args.vision_model, args.data_path))
    output = (args.output or ROOT / "artifacts" / "preflight" /
              f"vision_b{args.batch}_s{args.seq}_a{args.accumulation_steps}.json").resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite an existing probe report: {output}")
    if not output.is_relative_to(ROOT):
        raise SystemExit("Probe outputs must stay inside the evomind repository.")
    report = {"status": "checking", "probe_only": True, "quality_evaluation": False,
              "started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "argv": [sys.executable, *sys.argv], "torch_device": "cuda:0",
              "batch_size": args.batch, "max_seq_len": args.seq,
              "accumulation_steps": args.accumulation_steps, "effective_batch": args.batch * args.accumulation_steps,
              "hidden_size": 768, "num_hidden_layers": 8, "freeze_llm": 1,
              "dtype": "bfloat16", "learning_rate": 5e-6,
              "probe_updates": args.steps, "input_source": "bounded_real_official_single_image_batch",
              "text_weight": str(args.text_weight), "vision_model": str(args.vision_model),
              "data_path": str(args.data_path), "notes": [
                  "Full official model and real image preprocessing; no compressed model or cached encoder features.",
                  "Repeated fixed CPU batch; throughput includes host-to-device transfer and all model/optimizer work, but excludes disk read and CPU image preprocessing.",
                  "Probe optimizer updates are discarded; no checkpoint or evaluation score is produced.",
                  "OOM does not select or launch a replacement configuration."]}
    atomic_json(output, report)
    torch = None
    exit_code = 1
    try:
        for required in (args.text_weight, args.data_path, args.vision_model / "config.json",
                         args.vision_model / "preprocessor_config.json"):
            if not required.is_file():
                raise FileNotFoundError(f"Required probe input is missing: {required}")
        encoder_weights = sorted(args.vision_model.glob("*.safetensors"))
        if not encoder_weights:
            raise FileNotFoundError(f"No local safetensors encoder weights in {args.vision_model}")
        if not args.text_weight.name.endswith("_768.pth"):
            raise ValueError("Text weight must use the official *_768.pth filename contract.")
        report["nvidia_smi"] = command_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"])
        processes = command_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"])
        report["compute_processes_before"] = processes
        foreign_pids = [line for line in processes.splitlines() if line and line.split(",", 1)[0].strip() != str(os.getpid())]
        if foreign_pids:
            raise RuntimeError("Another compute process is active; no GPU probe started: " + "; ".join(foreign_pids))
        os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache" / "huggingface" / "datasets"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        sys.path.insert(0, str(VISION))
        import datasets  # Required before torch for the Windows Arrow DLL workaround.
        import torch as torch_module
        torch = torch_module
        from model.model_vlm import VLMConfig
        from trainer.trainer_utils import init_vlm_model, setup_seed, vlm_collate_fn
        from dataset.lm_dataset import VLMDataset
        import trainer.trainer_utils as vision_utils
        if not Path(vision_utils.__file__).resolve().is_relative_to(VISION):
            raise RuntimeError("Wrong trainer module resolved; run this script in a fresh Python process.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU fallback is intentionally disabled.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Official BF16 configuration is not supported on this GPU.")
        report.update(torch=torch.__version__, cuda=torch.version.cuda,
                      gpu_name=torch.cuda.get_device_name(0),
                      text_weight_sha256=digest(args.text_weight),
                      encoder_weight_hashes={path.name: digest(path) for path in encoder_weights},
                      code_commit=command_output(["git", "-C", str(VISION), "rev-parse", "HEAD"]),
                      code_hashes={str(path.relative_to(ROOT)): digest(path) for path in (
                          Path(__file__), VISION / "model" / "model_vlm.py", VISION / "model" / "model_minimind.py",
                          VISION / "trainer" / "trainer_utils.py", VISION / "dataset" / "lm_dataset.py")})
        setup_seed(42)
        torch.cuda.reset_peak_memory_stats(0)
        config = VLMConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=args.seq, use_moe=False)
        model, tokenizer, preprocess = init_vlm_model(
            config, from_weight=args.text_weight.stem[:-4], tokenizer_path=str(VISION / "model"),
            vision_model_path=str(args.vision_model), save_dir=str(args.text_weight.parent),
            device="cuda:0", freeze_llm=1)
        if model.vision_encoder is None or preprocess is None:
            raise RuntimeError("Official encoder loader returned None; local SigLIP weights/config are incompatible.")
        text_state = torch.load(args.text_weight, map_location="cpu", weights_only=True)
        target_state = model.state_dict()
        llm_keys = {key for key in target_state if not key.startswith(("vision_encoder.", "vision_proj."))}
        missing = sorted(llm_keys - set(text_state))
        unexpected = sorted(set(text_state) - llm_keys)
        mismatched = [key for key in llm_keys & set(text_state) if target_state[key].shape != text_state[key].shape]
        if missing or unexpected or mismatched:
            raise RuntimeError(f"Text checkpoint mismatch: missing={missing}, unexpected={unexpected}, shape={mismatched}")
        report["text_checkpoint_contract"] = {"llm_keys": len(llm_keys), "missing": missing, "unexpected": unexpected, "shape_mismatch": mismatched}
        del text_state, target_state
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        allowed = lambda name: name.startswith(("vision_proj.", "model.layers.0.", "model.layers.7."))
        if not trainable_names or any(not allowed(name) for name in trainable_names):
            raise RuntimeError("Freeze policy differs from projector + LLM first/last layer.")
        if any(parameter.requires_grad for parameter in model.vision_encoder.parameters()):
            raise RuntimeError("Vision encoder is not fully frozen.")
        report["parameters"] = {
            "total_with_vision_encoder": sum(parameter.numel() for parameter in model.parameters()),
            "non_encoder": sum(parameter.numel() for name, parameter in model.named_parameters() if not name.startswith("vision_encoder.")),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "trainable_names": trainable_names}
        rows, indices, dataset_rows = select_single_image_rows(args.data_path, args.batch)
        # Use all official tokenization, label masking, augmentation, image processing,
        # and collate methods; substitute only the bounded in-memory row storage.
        probe_dataset = VLMDataset.__new__(VLMDataset)
        probe_dataset.dataset = rows
        probe_dataset.tokenizer = tokenizer
        probe_dataset.max_length = args.seq
        probe_dataset.preprocess = preprocess
        probe_dataset.image_special_token = config.image_special_token * config.image_token_len
        probe_dataset.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        probe_dataset.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        cpu_batch = vlm_collate_fn([probe_dataset[index] for index in range(args.batch)])
        if cpu_batch[0].shape != (args.batch, args.seq):
            raise RuntimeError(f"Unexpected token batch shape: {cpu_batch[0].shape}")
        marker_counts = (cpu_batch[0] == config.image_ids[0]).sum(dim=1)
        supervised_counts = (cpu_batch[1][:, 1:] != -100).sum(dim=1)
        if not torch.all(marker_counts == config.image_token_len) or not torch.all(supervised_counts > 0):
            raise RuntimeError("Selected data was truncated into missing visual tokens or empty assistant labels.")
        report["inputs"] = {"dataset_rows": dataset_rows, "row_indices": indices,
                            "image_markers_per_sample": marker_counts.tolist(),
                            "supervised_tokens_per_sample": supervised_counts.tolist(),
                            "pixel_shapes": {key: list(value.shape) for key, value in cpu_batch[2].items()}}
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
        # init_vlm_model constructs train-mode LLM/projector and explicitly sets the
        # frozen encoder to eval. Preserve these modes, as the SFT entry point does.
        report["vision_encoder_training_mode"] = model.vision_encoder.training
        free, total = torch.cuda.mem_get_info(0)
        report["before_steps"] = {"free_mib": free / 2**20, "total_mib": total / 2**20,
                                  "allocated_mib": torch.cuda.memory_allocated(0) / 2**20}
        report["status"] = "running"
        report["iterations"] = []
        atomic_json(output, report)
        for update in range(args.steps):
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            loss_value = None
            for _ in range(args.accumulation_steps):
                input_ids, labels = (value.to("cuda:0") for value in cpu_batch[:2])
                pixels = {key: value.to("cuda:0") for key, value in cpu_batch[2].items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(input_ids, labels=labels, pixel_values=pixels)
                    full_loss = result.loss + result.aux_loss
                    loss = full_loss / args.accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite vision probe loss; stopping without retry.")
                loss.backward()
                loss_value = full_loss.item()
                del input_ids, labels, pixels, result, loss, full_loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite vision probe gradient norm; stopping without retry.")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter() - started
            iteration = {"optimizer_update": update + 1, "warmup": update == 0,
                         "seconds": elapsed, "diagnostic_last_microbatch_loss": loss_value,
                         "examples_per_second": args.batch * args.accumulation_steps / elapsed,
                         "padded_tokens_per_second": args.batch * args.accumulation_steps * args.seq / elapsed}
            report["iterations"].append(iteration)
            report.update(peak_allocated_mib=torch.cuda.max_memory_allocated(0) / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved(0) / 2**20)
            atomic_json(output, report)
            print("VISION_PROBE_ONLY " + json.dumps(iteration), flush=True)
        measured = report["iterations"][1:]
        seconds = sum(item["seconds"] for item in measured)
        report["measured_examples_per_second"] = len(measured) * args.batch * args.accumulation_steps / seconds
        report.update(status="success", measured_optimizer_updates=len(measured),
                      finished_at=datetime.now(timezone.utc).isoformat())
        exit_code = 0
    except Exception as error:
        oom = torch is not None and isinstance(error, torch.cuda.OutOfMemoryError)
        report.update(status="oom" if oom else "failed", error=str(error), traceback=traceback.format_exc(),
                      finished_at=datetime.now(timezone.utc).isoformat())
        if torch is not None and torch.cuda.is_initialized():
            report.update(peak_allocated_mib=torch.cuda.max_memory_allocated(0) / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved(0) / 2**20)
        exit_code = 3 if oom else 1
        print(f"VISION_PROBE_{report['status'].upper()}: {error}; no fallback was launched.", file=sys.stderr, flush=True)
    finally:
        report["exit_code"] = exit_code
        atomic_json(output, report)
        print(f"VISION_PROBE_REPORT {output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
