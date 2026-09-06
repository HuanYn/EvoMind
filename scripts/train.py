import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig
from model.data import NextTokenDataset
from model.model_minimind import MiniMindModel
from model.tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/minimind_char.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer.load(args.tokenizer)
    token_ids = tokenizer.encode(args.text.read_text(encoding="utf-8"), add_bos=True, add_eos=True)
    dataset = NextTokenDataset(token_ids, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, dim=args.dim, n_layers=args.layers, n_heads=args.heads, n_kv_heads=args.kv_heads, hidden_dim=args.hidden_dim, max_seq_len=args.seq_len)
    model = MiniMindModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    print(f"device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch:03d} | loss: {total_loss / len(loader):.4f}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "model_state_dict": model.state_dict()}, args.checkpoint)
    print(f"saved checkpoint to: {args.checkpoint}")


if __name__ == "__main__":
    main()
