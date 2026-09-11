"""Plot an explicitly labelled GRPO/CISPO rollout-reuse mechanism ablation.

This reads append-only runtime JSONL files; it neither trains nor selects a
product checkpoint.  Before the first update the idealized ratio is 1; actual
evaluation can differ numerically. Later replays show whether GRPO's narrow clip or CISPO's wide
importance cap actually intervened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if "rollout_reuse_index" in value:
            rows.append(value)
    if not rows:
        raise ValueError(f"No rollout-reuse metrics in {path}")
    return rows


def series(rows, key):
    return [row["optimizer_update"] for row in rows], [row.get(key, float("nan")) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cispo", required=True, type=Path, help="CISPO metrics.jsonl")
    parser.add_argument("--grpo", required=True, type=Path, help="GRPO metrics.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="PNG output path")
    args = parser.parse_args()
    runs = {"CISPO": load(args.cispo), "GRPO": load(args.grpo)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    charts = (
        ("ratio_p95", "Per-token importance ratio (p95)"),
        ("ratio_max", "Per-token importance ratio (max)"),
        ("native_intervention_rate", "Native intervention rate"),
        ("kl_penalty", "Reference KL penalty"),
    )
    for axis, (key, title) in zip(axes.flat, charts):
        for name, rows in runs.items():
            x, y = series(rows, key)
            axis.plot(x, y, linewidth=1.4, label=name)
        axis.set_title(title)
        axis.set_xlabel("Optimizer update")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Mechanism ablation: K=4 updates per frozen rollout (not a product benchmark)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
