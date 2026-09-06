"""Inspect expert utilization on a fixed set of token blocks."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig
from model.data import NextTokenDataset
from model.model_minimind import MiniMindModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MiniMind MoE expert routing")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    if not config.use_moe:
        raise ValueError("This checkpoint is dense; it does not contain MoE routing")

    payload = torch.load(args.data, map_location="cpu", weights_only=True)
    loader = DataLoader(
        NextTokenDataset(payload["token_ids"].long(), args.seq_len),
        batch_size=args.batch_size,
    )
    model = MiniMindModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    load_sum = torch.zeros(config.n_layers, config.num_experts, device=device)
    aux_loss_sum = 0.0
    batches = 0
    with torch.no_grad():
        for input_ids, _ in loader:
            if batches >= args.max_batches:
                break
            _, router_aux_loss, expert_fractions = model(
                input_ids.to(device), return_router_loss=True
            )
            load_sum += torch.stack(expert_fractions)
            aux_loss_sum += router_aux_loss.item()
            batches += 1

    mean_load = load_sum / batches
    print(f"device: {device} | checkpoint step: {checkpoint['step']} | batches: {batches}")
    print(f"mean router auxiliary loss: {aux_loss_sum / batches:.6f}")
    for layer_index, load in enumerate(mean_load.cpu()):
        formatted = ", ".join(f"expert {expert}: {value:.2%}" for expert, value in enumerate(load))
        print(f"layer {layer_index}: {formatted}")


if __name__ == "__main__":
    main()
