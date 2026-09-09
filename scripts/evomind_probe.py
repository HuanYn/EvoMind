"""A synthetic GPU memory preflight; these losses are NOT evaluation results."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import datasets  # Windows pyarrow/torch import order
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--steps", type=int, default=3)
    a = p.parse_args()
    torch.manual_seed(42)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8)).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ids = torch.randint(4, 6400, (a.batch, a.seq), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(a.steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(ids, labels=ids)
            loss = output.loss + output.aux_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite preflight loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter()-t)
        print(f"PROBE_ONLY b={a.batch} seq={a.seq} iteration={i+1} seconds={times[-1]:.3f} peakMiB={torch.cuda.max_memory_allocated()/2**20:.1f}", flush=True)
    result = {"synthetic_preflight_only": True, "batch": a.batch, "seq": a.seq,
              "parameters": sum(p.numel() for p in model.parameters()), "seconds": times,
              "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
              "device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    path = ROOT / "artifacts" / "preflight" / f"b{a.batch}_s{a.seq}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
