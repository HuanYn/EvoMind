"""Load the frozen RLAIF reward model and verify score discrimination."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainer.train_grpo import MiniMindRewardModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the frozen MiniMind RLAIF reward model")
    parser.add_argument("--model", default="models/internlm2-1_8b-reward")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    path = Path(args.model)
    if not path.exists():
        raise FileNotFoundError(path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    reward_model = MiniMindRewardModel(str(path), device, dtype)
    messages = [{"role": "user", "content": "请用简洁中文解释 MoE 的核心作用。"}]
    candidates = {
        "specific": "MoE 用多个前馈专家替代一个稠密前馈层，路由器为每个 token 只选择少数专家。这样总参数可以增大，但每个 token 的激活计算量不必随专家总数线性增加。",
        "vague": "MoE 是一种非常重要的人工智能技术。MoE 有很多专家，专家可以帮助模型变得更好。",
    }
    print(f"device: {device} | dtype: {dtype} | model: {path}")
    for name, response in candidates.items():
        score = reward_model.score(messages, response)
        print(f"{name}: {score:.4f}")
    if device.type == "cuda":
        print(f"peak allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
