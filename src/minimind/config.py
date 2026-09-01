from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 256
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    hidden_dim: int = 256
    max_seq_len: int = 128
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads