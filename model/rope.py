import torch


def precompute_rope_frequencies(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float)
    frequencies = torch.outer(positions, inv_freq)
    return frequencies.cos(), frequencies.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x shaped (batch, time, heads, head_dim)."""
    cos = cos.index_select(0, positions)[None, :, None, :]
    sin = sin.index_select(0, positions)[None, :, None, :]
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)
