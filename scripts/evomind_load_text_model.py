"""Shared strict local HF-compatible model/adapter loader for evomind evaluation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]


def load_model(checkpoint, lora=None, *, device="cuda", use_moe=False):
    sys.path.insert(0, str(ROOT))
    import datasets  # noqa: F401 -- Windows DLL order
    import torch
    from transformers import AutoTokenizer
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=use_moe))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
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
