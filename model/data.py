import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    def __init__(self, token_ids: list[int] | torch.Tensor, seq_len: int):
        if len(token_ids) <= seq_len:
            raise ValueError("token stream must contain at least seq_len + 1 tokens")
        self.tokens = torch.as_tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.num_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        start = index * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]
