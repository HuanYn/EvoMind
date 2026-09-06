"""Benchmark Dense and MoE MiniMind inference under the same protocol."""

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig
from model.model_minimind import MiniMindModel


MIB = 1024**2


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile without extra dependencies."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(latencies_ms),
        "p10_ms": percentile(latencies_ms, 0.10),
        "p90_ms": percentile(latencies_ms, 0.90),
    }


def peak_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / MIB


def peak_reserved_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_reserved(device) / MIB


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def timed_call(function, device: torch.device) -> tuple[object, float]:
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


def model_signature(config: ModelConfig) -> dict:
    """Fields that must match for a controlled Dense-versus-MoE comparison."""
    return {
        "vocab_size": config.vocab_size,
        "dim": config.dim,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "hidden_dim": config.hidden_dim,
        "max_seq_len": config.max_seq_len,
        "flash_attn": config.flash_attn,
        "tie_word_embeddings": config.tie_word_embeddings,
    }


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MiniMindModel, ModelConfig, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = MiniMindModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    step = int(checkpoint["step"])
    del checkpoint
    model = model.to(device=device, dtype=dtype).eval()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, config, step


@torch.inference_mode()
def prefill_once(model: MiniMindModel, input_ids: torch.Tensor):
    return model(input_ids, use_cache=True)


@torch.inference_mode()
def prepare_decode(model: MiniMindModel, input_ids: torch.Tensor):
    _, past_key_values = model(input_ids, use_cache=True)
    return past_key_values


@torch.inference_mode()
def decode_steps(
    model: MiniMindModel,
    continuation_ids: torch.Tensor,
    past_key_values,
) -> None:
    # Teacher-force the same validation tokens through both models so sampling
    # and different generated text cannot distort the speed comparison.
    for step in range(continuation_ids.size(1)):
        _, past_key_values = model(
            continuation_ids[:, step : step + 1],
            past_key_values=past_key_values,
            use_cache=True,
        )


def benchmark_prefill(
    model: MiniMindModel,
    input_ids: torch.Tensor,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict:
    for _ in range(warmup):
        result = prefill_once(model, input_ids)
        del result
    synchronize(device)

    latencies_ms = []
    peaks_mib = []
    reserved_peaks_mib = []
    for _ in range(repeats):
        reset_peak_memory(device)
        result, elapsed_ms = timed_call(lambda: prefill_once(model, input_ids), device)
        latencies_ms.append(elapsed_ms)
        peak = peak_memory_mib(device)
        reserved_peak = peak_reserved_memory_mib(device)
        if peak is not None:
            peaks_mib.append(peak)
        if reserved_peak is not None:
            reserved_peaks_mib.append(reserved_peak)
        del result

    summary = latency_summary(latencies_ms)
    summary["tokens_per_second"] = input_ids.numel() / (summary["median_ms"] / 1000.0)
    summary["peak_memory_mib"] = max(peaks_mib) if peaks_mib else None
    summary["peak_reserved_memory_mib"] = (
        max(reserved_peaks_mib) if reserved_peaks_mib else None
    )
    return summary


def benchmark_decode(
    model: MiniMindModel,
    input_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict:
    for _ in range(warmup):
        cache = prepare_decode(model, input_ids)
        decode_steps(model, continuation_ids, cache)
        del cache
    synchronize(device)

    latencies_ms = []
    peaks_mib = []
    reserved_peaks_mib = []
    for _ in range(repeats):
        cache = prepare_decode(model, input_ids)
        synchronize(device)
        reset_peak_memory(device)
        _, elapsed_ms = timed_call(
            lambda: decode_steps(model, continuation_ids, cache),
            device,
        )
        latencies_ms.append(elapsed_ms)
        peak = peak_memory_mib(device)
        reserved_peak = peak_reserved_memory_mib(device)
        if peak is not None:
            peaks_mib.append(peak)
        if reserved_peak is not None:
            reserved_peaks_mib.append(reserved_peak)
        del cache

    summary = latency_summary(latencies_ms)
    number_of_steps = continuation_ids.size(1)
    summary["time_per_output_token_ms"] = summary["median_ms"] / number_of_steps
    summary["tokens_per_second"] = (
        input_ids.size(0) * number_of_steps / (summary["median_ms"] / 1000.0)
    )
    summary["peak_memory_mib"] = max(peaks_mib) if peaks_mib else None
    summary["peak_reserved_memory_mib"] = (
        max(reserved_peaks_mib) if reserved_peaks_mib else None
    )
    return summary


def benchmark_checkpoint(
    label: str,
    checkpoint_path: Path,
    input_ids_cpu: torch.Tensor,
    continuation_ids_cpu: torch.Tensor,
    expected_vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    number_of_decode_steps: int,
) -> tuple[dict, dict]:
    model, config, step = load_model(checkpoint_path, device, dtype)
    expected_moe = label == "moe"
    if config.use_moe != expected_moe:
        raise ValueError(
            f"{label}: checkpoint use_moe={config.use_moe}; check that Dense and MoE paths were not swapped"
        )
    if config.vocab_size != expected_vocab_size:
        raise ValueError(
            f"{label}: checkpoint vocabulary size {config.vocab_size} does not match "
            f"data vocabulary size {expected_vocab_size}"
        )
    if input_ids_cpu.size(1) + number_of_decode_steps > config.max_seq_len:
        raise ValueError(
            f"{label}: prompt length + decode steps exceeds max_seq_len={config.max_seq_len}"
        )
    input_ids = input_ids_cpu.to(device)
    continuation_ids = continuation_ids_cpu.to(device)
    synchronize(device)
    model_memory_mib = (
        torch.cuda.memory_allocated(device) / MIB if device.type == "cuda" else None
    )

    result = {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": step,
        "config": asdict(config),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_and_input_memory_mib": model_memory_mib,
        "prefill": benchmark_prefill(model, input_ids, warmup, repeats, device),
        "decode": benchmark_decode(
            model,
            input_ids,
            continuation_ids,
            warmup,
            repeats,
            device,
        ),
    }
    signature = model_signature(config)
    del model, input_ids, continuation_ids
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, signature


def parse_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[name]
    if device.type == "cpu" and dtype != torch.float32:
        print(f"warning: {name} requested on CPU; falling back to float32")
        return torch.float32
    return dtype


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Dense and MoE MiniMind prefill/decode efficiency"
    )
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--moe-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Prepared validation token stream")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/inference_dense_vs_moe.json"),
    )
    args = parser.parse_args()

    for name in ("batch_size", "prompt_len", "decode_steps", "repeats"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = parse_dtype(args.dtype, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    data_payload = torch.load(args.data, map_location="cpu", weights_only=True)
    token_ids = data_payload["token_ids"].long()
    sequence_length = args.prompt_len + args.decode_steps
    needed_tokens = args.batch_size * sequence_length
    if token_ids.numel() < needed_tokens:
        raise ValueError(f"Need {needed_tokens} input tokens, but data has {token_ids.numel()}")
    benchmark_ids = token_ids[:needed_tokens].reshape(args.batch_size, sequence_length)
    input_ids = benchmark_ids[:, : args.prompt_len].clone()
    continuation_ids = benchmark_ids[:, args.prompt_len :].clone()

    print(
        f"device: {device} | dtype: {str(dtype).removeprefix('torch.')} | "
        f"batch: {args.batch_size} | prompt: {args.prompt_len} | decode steps: {args.decode_steps}"
    )
    results = []
    signatures = []
    for label, path in (
        ("dense", args.dense_checkpoint),
        ("moe", args.moe_checkpoint),
    ):
        print(f"benchmarking {label}: {path}")
        result, signature = benchmark_checkpoint(
            label,
            path,
            input_ids,
            continuation_ids,
            int(data_payload["vocab_size"]),
            device,
            dtype,
            args.warmup,
            args.repeats,
            args.decode_steps,
        )
        results.append(result)
        signatures.append(signature)
        print(
            f"{label}: prefill {result['prefill']['median_ms']:.2f} ms / "
            f"{result['prefill']['tokens_per_second']:.1f} tok/s | "
            f"decode {result['decode']['time_per_output_token_ms']:.2f} ms/token / "
            f"{result['decode']['tokens_per_second']:.1f} tok/s | "
            f"peak {max(result['prefill']['peak_memory_mib'] or 0, result['decode']['peak_memory_mib'] or 0):.1f} MiB"
        )

    if signatures[0] != signatures[1]:
        raise ValueError(
            "Dense and MoE base architectures do not match; this is not a controlled comparison"
        )

    report = {
        "protocol": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "dtype": str(dtype).removeprefix("torch."),
            "batch_size": args.batch_size,
            "prompt_length": args.prompt_len,
            "decode_steps": args.decode_steps,
            "warmup_runs": args.warmup,
            "measured_runs": args.repeats,
            "kv_cache": True,
            "input_data": str(args.data),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved report: {args.output}")


if __name__ == "__main__":
    main()
