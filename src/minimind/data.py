import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int):
        if len(token_ids) <= seq_len:
            raise ValueError("token stream must contain at least seq_len + 1 tokens")
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.starts = list(range(0, len(self.tokens) - seq_len, seq_len))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        chunk = self.tokens[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]
