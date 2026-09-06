"""Controlled repetition diagnostics for pretraining and SFT checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate import sample  # noqa: E402
from trainer.train_full_sft import sha256_file  # noqa: E402
from model.chat_template import TEMPLATE_VERSION, encode_generation_prompt  # noqa: E402
from model.config import ModelConfig  # noqa: E402
from model.model_minimind import MiniMindModel  # noqa: E402
from model.tokenizer import BPETokenizer  # noqa: E402


DEFAULT_DECODING_CONFIGS = {
    "greedy": {"temperature": 0.0, "top_k": 0, "top_p": 1.0},
    "sampled": {"temperature": 0.7, "top_k": 40, "top_p": 0.9},
}


def _parse_sampling_config(value: str) -> tuple[str, dict[str, float | int]]:
    """Parse ``name,temperature,top_k,top_p`` for a sampled decoding run."""
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "sampling config must be name,temperature,top_k,top_p"
        )
    name = parts[0].strip()
    try:
        temperature = float(parts[1])
        top_k = int(parts[2])
        top_p = float(parts[3])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "temperature/top_k/top_p must be numeric"
        ) from error
    if not name or name == "greedy" or temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise argparse.ArgumentTypeError("invalid sampled decoding configuration")
    return name, {"temperature": temperature, "top_k": top_k, "top_p": top_p}


def _atomic_text_save(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use LABEL=PATH syntax")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(
            "checkpoint label must be non-empty and checkpoint path must exist"
        )
    return label, path


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    seen_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            prompt_id = record.get("id")
            messages = record.get("messages")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(f"prompt line {line_number} has no non-empty id")
            if prompt_id in seen_ids:
                raise ValueError(f"duplicate prompt id: {prompt_id}")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"prompt {prompt_id} has no messages list")
            seen_ids.add(prompt_id)
            prompts.append(record)
    if not prompts:
        raise ValueError("prompt suite is empty")
    return prompts


def _repetition_metrics(ids: list[int], ngram_size: int = 4) -> dict[str, float | int]:
    if not ids:
        return {
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "repeated_4gram_fraction": 0.0,
            "max_identical_token_run": 0,
        }
    unigrams = len(set(ids)) / len(ids)
    bigrams = list(zip(ids, ids[1:]))
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 1.0
    ngrams = [
        tuple(ids[index : index + ngram_size])
        for index in range(len(ids) - ngram_size + 1)
    ]
    repeated_ngrams = (
        (len(ngrams) - len(set(ngrams))) / len(ngrams) if ngrams else 0.0
    )
    max_run = 1
    current_run = 1
    for previous, current in zip(ids, ids[1:]):
        current_run = current_run + 1 if previous == current else 1
        max_run = max(max_run, current_run)
    return {
        "distinct_1": unigrams,
        "distinct_2": distinct_2,
        "repeated_4gram_fraction": repeated_ngrams,
        "max_identical_token_run": max_run,
    }


@torch.inference_mode()
def _generate(
    model: MiniMindModel,
    prompt_ids: list[int],
    tokenizer: BPETokenizer,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[list[int], bool]:
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids: list[int] = []
    past_key_values = None
    terminated_with_eos = False
    for _ in range(max_new_tokens):
        logits, past_key_values = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = sample(logits[:, -1, :], temperature, top_k, top_p)
        next_id = int(next_token.item())
        if next_id == tokenizer.eos_id:
            terminated_with_eos = True
            break
        generated_ids.append(next_id)
        if len(prompt_ids) + len(generated_ids) >= model.config.max_seq_len:
            break
        input_ids = next_token
    return generated_ids, terminated_with_eos


def _load_model(path: Path, device: torch.device) -> tuple[MiniMindModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is missing config or model_state_dict")
    config = ModelConfig(**checkpoint["config"])
    model = MiniMindModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    metadata = {
        "stage": checkpoint.get("stage", "pretrain_legacy"),
        "step": int(checkpoint.get("step", 0)),
        "config": config,
    }
    return model, metadata


def _mean(records: list[dict[str, Any]], key: str) -> float:
    return sum(float(record[key]) for record in records) / len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose repetition with fixed prompts and controlled decoding"
    )
    parser.add_argument(
        "--checkpoints",
        type=_parse_checkpoint,
        nargs="+",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("eval_assets/repetition_prompts_zh_v1.jsonl"),
    )
    parser.add_argument("--system", default="你是一个严谨、简洁的 AI 助手。")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--open-thinking", action="store_true")
    parser.add_argument(
        "--sampling-config",
        action="append",
        type=_parse_sampling_config,
        metavar="NAME,TEMP,TOP_K,TOP_P",
        help=(
            "Sampled decoding configuration; repeat for a grid. When omitted, "
            "uses the baseline sampled=0.7,40,0.9."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/repetition_diagnosis.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/evaluation/repetition_diagnosis_summary.json"),
    )
    args = parser.parse_args()
    if not args.tokenizer.is_file():
        raise FileNotFoundError(args.tokenizer)
    if not args.prompts.is_file():
        raise FileNotFoundError(args.prompts)
    if args.max_length < 1 or args.max_new_tokens < 1:
        raise ValueError("max-length and max-new-tokens must be positive")
    labels = [label for label, _ in args.checkpoints]
    if len(set(labels)) != len(labels):
        raise ValueError("checkpoint labels must be unique")
    sampling_configs = args.sampling_config or [
        ("sampled", DEFAULT_DECODING_CONFIGS["sampled"])
    ]
    sample_names = [name for name, _ in sampling_configs]
    if len(set(sample_names)) != len(sample_names):
        raise ValueError("sampled decoding configuration names must be unique")
    args.decoding_configs = {
        "greedy": DEFAULT_DECODING_CONFIGS["greedy"],
        **dict(sampling_configs),
    }
    return args


def main() -> None:
    args = parse_args()
    tokenizer = BPETokenizer.load(args.tokenizer)
    prompts = _load_prompts(args.prompts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records: list[dict[str, Any]] = []

    for label, checkpoint_path in args.checkpoints:
        model, metadata = _load_model(checkpoint_path, device)
        config: ModelConfig = metadata["config"]
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError(f"{label}: tokenizer vocabulary does not match checkpoint")
        if args.max_length > config.max_seq_len:
            raise ValueError(f"{label}: max-length exceeds checkpoint max_seq_len")

        for prompt_index, prompt in enumerate(prompts):
            conversations = []
            if args.system:
                conversations.append({"role": "system", "content": args.system})
            conversations.extend(prompt["messages"])
            prompt_ids = encode_generation_prompt(
                conversations,
                tokenizer,
                max_length=args.max_length,
                open_thinking=args.open_thinking,
            )
            for decoding_name, decoding in args.decoding_configs.items():
                seed = args.seed + prompt_index
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                generated_ids, terminated_with_eos = _generate(
                    model,
                    prompt_ids,
                    tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    **decoding,
                )
                records.append(
                    {
                        "schema_version": 1,
                        "checkpoint_label": label,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_stage": metadata["stage"],
                        "checkpoint_step": metadata["step"],
                        "prompt_suite": str(args.prompts),
                        "prompt_id": prompt["id"],
                        "category": prompt.get("category"),
                        "prompt_token_count": len(prompt_ids),
                        "decoding": {"name": decoding_name, **decoding, "seed": seed},
                        "generated_token_count": len(generated_ids),
                        "terminated_with_eos": terminated_with_eos,
                        "repetition": _repetition_metrics(generated_ids),
                        "response": tokenizer.decode(generated_ids),
                    }
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    lines = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    _atomic_text_save(lines, args.output)
    summary_rows = []
    for label, _ in args.checkpoints:
        for decoding_name in args.decoding_configs:
            group = [
                record
                for record in records
                if record["checkpoint_label"] == label
                and record["decoding"]["name"] == decoding_name
            ]
            repetition = [record["repetition"] for record in group]
            summary_rows.append(
                {
                    "checkpoint_label": label,
                    "decoding": decoding_name,
                    "prompts": len(group),
                    "mean_generated_tokens": _mean(group, "generated_token_count"),
                    "eos_termination_rate": _mean(group, "terminated_with_eos"),
                    "mean_distinct_1": _mean(repetition, "distinct_1"),
                    "mean_distinct_2": _mean(repetition, "distinct_2"),
                    "mean_repeated_4gram_fraction": _mean(
                        repetition,
                        "repeated_4gram_fraction",
                    ),
                    "mean_max_identical_token_run": _mean(
                        repetition,
                        "max_identical_token_run",
                    ),
                }
            )
    summary = {
        "schema_version": 1,
        "purpose": "Token-level repetition proxies, not answer quality scores.",
        "device": str(device),
        "template_version": TEMPLATE_VERSION,
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "prompts": str(args.prompts),
        "prompts_sha256": sha256_file(args.prompts),
        "system": args.system,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "open_thinking": args.open_thinking,
        "decoding_configs": args.decoding_configs,
        "rows": summary_rows,
    }
    _atomic_text_save(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        args.summary,
    )
    print(f"device: {device} | prompts: {len(prompts)} | records: {len(records)}")
    for row in summary_rows:
        print(
            f"{row['checkpoint_label']} / {row['decoding']}: "
            f"distinct-2 {row['mean_distinct_2']:.3f} | "
            f"repeat-4 {row['mean_repeated_4gram_fraction']:.3f} | "
            f"EOS {row['eos_termination_rate']:.1%}"
        )
    print(f"saved records: {args.output}")
    print(f"saved summary: {args.summary}")


if __name__ == "__main__":
    main()
