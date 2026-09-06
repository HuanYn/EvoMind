from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 16_000
    dim: int = 768
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    hidden_dim: int = 2432
    max_seq_len: int = 32_768
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    tie_word_embeddings: bool = True
    flash_attn: bool = True
    use_moe: bool = False
    num_experts: int = 4
    num_experts_per_tok: int = 1
    router_aux_loss_coef: float = 5e-4

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads
