import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig
from model.model_minimind import MiniMindModel
from model.tokenizer import BPETokenizer


def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
        remove[..., 0] = False
        logits = logits.scatter(-1, sorted_indices, sorted_logits.masked_fill(remove, float("-inf")))
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", default="人工智能")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = MiniMindModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = BPETokenizer.load(args.tokenizer)
    ids = tokenizer.encode(args.prompt, add_bos=True)
    if len(ids) > config.max_seq_len:
        ids = ids[-config.max_seq_len :]
    torch.manual_seed(args.seed)

    with torch.no_grad():
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        past_key_values = None
        for _ in range(args.max_new_tokens):
            logits, past_key_values = model(
                input_ids, past_key_values=past_key_values, use_cache=True
            )
            next_token = sample(logits[:, -1, :], args.temperature, args.top_k, args.top_p)
            next_id = next_token.item()
            ids.append(next_id)
            if next_id == tokenizer.eos_id:
                break
            if len(ids) >= config.max_seq_len:
                break
            input_ids = next_token
    print(f"device: {device} | checkpoint step: {checkpoint['step']}")
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
