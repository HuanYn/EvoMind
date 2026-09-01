import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.minimind.config import ModelConfig
from src.minimind.model import MiniMindModel
from src.minimind.tokenizer import CharTokenizer


def sample(logits, temperature, top_k):
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", default="MiniMind")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = MiniMindModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = CharTokenizer.load(args.tokenizer)
    ids = tokenizer.encode(args.prompt, add_bos=True)

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            input_ids = torch.tensor([ids[-config.max_seq_len:]], dtype=torch.long, device=device)
            next_id = sample(model(input_ids)[:, -1, :], args.temperature, args.top_k).item()
            ids.append(next_id)
            if next_id == tokenizer.eos_id:
                break
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
