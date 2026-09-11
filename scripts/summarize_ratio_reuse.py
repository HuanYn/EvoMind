"""Summarize an explicitly non-product GRPO/CISPO rollout-reuse ablation."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean


FIELDS = (
    "ratio_mean", "ratio_p95", "ratio_max", "grpo_suppressed_rate",
    "cispo_capped_rate", "native_intervention_rate", "kl_penalty", "reward",
)


def load(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "rollout_reuse_index" in row:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rollout-reuse rows: {path}")
    return rows


def average(rows):
    return {field: fmean(float(row[field]) for row in rows) for field in FIELDS}


def summarize(name: str, rows):
    by_replay = defaultdict(list)
    for row in rows:
        by_replay[int(row["rollout_reuse_index"])].append(row)
    return {
        "loss_type": name,
        "updates": len(rows),
        "replay_total": int(rows[0]["rollout_reuse_total"]),
        "overall_mean": average(rows),
        "by_replay_index": {str(index): {"updates": len(group), "mean": average(group)}
                            for index, group in sorted(by_replay.items())},
        "interpretation": {
            "grpo_suppressed_rate": "Fraction meeting GRPO's directional 1±epsilon rule; native for GRPO, counterfactual for CISPO.",
            "cispo_capped_rate": "Fraction beyond CISPO's wide epsilon_high cap.",
            "native_intervention_rate": "The intervention rate of the loss actually optimized in this run.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cispo", required=True, type=Path)
    parser.add_argument("--grpo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "scope": "mechanism ablation only; not a product quality benchmark",
        "cispo": summarize("cispo", load(args.cispo)),
        "grpo": summarize("grpo", load(args.grpo)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
