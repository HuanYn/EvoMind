import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        hidden = self.w1(x)
        hidden = F.relu(hidden)
        return self.w2(hidden)