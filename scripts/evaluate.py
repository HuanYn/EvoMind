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

from src.minimind.config import ModelConfig
from src.minimind.data import NextTokenDataset
from src.minimind.model import MiniMindModel
from src.minimind.tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = MiniMindModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = CharTokenizer.load(args.tokenizer)
    token_ids = tokenizer.encode(args.text.read_text(encoding="utf-8"), add_bos=True, add_eos=True)
    loader = DataLoader(NextTokenDataset(token_ids, config.max_seq_len), batch_size=args.batch_size)

    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")
            total_nll += nll.item()
            total_tokens += labels.numel()

    loss = total_nll / total_tokens
    print(f"tokens: {total_tokens}")
    print(f"cross entropy: {loss:.4f}")
    print(f"perplexity: {math.exp(min(loss, 20)):.4f}")


if __name__ == "__main__":
    main()
