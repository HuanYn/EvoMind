"""Shared strict local HF-compatible model/adapter loader for evomind evaluation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]


def extract_model_state(checkpoint):
    """Read either a pure state dict or an EvoMind training checkpoint.

    Training checkpoints retain optimizer/epoch metadata under the outer
    dictionary and store the actual model weights under ``model``.  We accept
    that documented wrapper, but still hand the resulting state dict to
    ``load_state_dict(..., strict=True)`` so a wrong architecture never loads
    partially or silently.
    """
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model") if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain a non-empty model state dict")
    if not all(isinstance(name, str) and torch.is_tensor(value) for name, value in state.items()):
        raise ValueError("checkpoint model field is not a tensor state dict")
    return state


def load_model(checkpoint, lora=None, *, device="cuda", use_moe=False):
    sys.path.insert(0, str(ROOT))
    import datasets  # noqa: F401 -- Windows DLL order
    import torch
    from transformers import AutoTokenizer
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=use_moe))
    model.load_state_dict(extract_model_state(checkpoint), strict=True)
    if lora:
        from model.model_lora import apply_lora
        apply_lora(model)
        adapter = torch.load(lora, map_location="cpu", weights_only=True)
        expected = {k for k in model.state_dict() if ".lora." in k}
        if set(adapter) != expected:
            raise ValueError("LoRA adapter missing/unexpected keys; refusing partial adaptation")
        combined = model.state_dict()
        combined.update(adapter)
        model.load_state_dict(combined, strict=True)
    return model.to(device).eval().requires_grad_(False), tokenizer
