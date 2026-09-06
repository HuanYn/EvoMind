import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig
from model.data import NextTokenDataset
from model.model_minimind import MiniMindModel
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Prepared .pt token stream")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=0, help="0 evaluates the full stream")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = MiniMindModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    payload = torch.load(args.data, map_location="cpu", weights_only=True)
    token_ids = payload["token_ids"].long()
    if payload["vocab_size"] != config.vocab_size:
        raise ValueError("Checkpoint and evaluation token stream use different vocabulary sizes")
    loader = DataLoader(NextTokenDataset(token_ids, args.seq_len), batch_size=args.batch_size)

    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch_index, (input_ids, labels) in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")
            total_nll += nll.item()
            total_tokens += labels.numel()

    loss = total_nll / total_tokens
    print(f"device: {device} | checkpoint step: {checkpoint['step']}")
    print(f"tokens: {total_tokens}")
    print(f"cross entropy: {loss:.4f}")
    print(f"perplexity: {math.exp(min(loss, 20)):.4f}")


if __name__ == "__main__":
    main()
