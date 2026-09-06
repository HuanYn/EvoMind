"""Plot one or more JSONL v1 training histories as a 2x3 PNG dashboard."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.training_history import deduplicate_records, load_training_history  # noqa: E402


def _load_pyplot():
    """Import matplotlib only when plotting is requested, using a headless backend."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _finite_number(value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        return math.nan
    return number


def _series(
    records: Sequence[Mapping[str, Any]],
    field: str,
    *,
    positive: bool = False,
) -> tuple[list[int], list[float]]:
    return (
        [int(record["step"]) for record in records],
        [_finite_number(record.get(field), positive=positive) for record in records],
    )


def _has_finite(values: Iterable[float]) -> bool:
    return any(math.isfinite(value) for value in values)


def _smooth_finite_segments(values: Sequence[float], window: int) -> list[float]:
    """Trailing mean that resets at each missing/invalid metric.

    Resetting is important: a gap means the metric was not observed, so values
    on opposite sides must never be averaged together or used to fill the gap.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("smoothing_window must be a positive integer")
    if window == 1:
        return list(values)
    smoothed: list[float] = []
    segment: list[float] = []
    for value in values:
        if not math.isfinite(value):
            segment.clear()
            smoothed.append(math.nan)
            continue
        segment.append(value)
        if len(segment) > window:
            del segment[0]
        smoothed.append(sum(segment) / len(segment))
    return smoothed


def _router_max_load(record: Mapping[str, Any]) -> float:
    direct = _finite_number(record.get("router_max_load"))
    if math.isfinite(direct):
        return direct
    by_layer = record.get("router_max_load_by_layer")
    if not isinstance(by_layer, list):
        return math.nan
    finite = [_finite_number(value) for value in by_layer]
    finite = [value for value in finite if math.isfinite(value)]
    return max(finite) if finite else math.nan


def _legend_if_present(axis, **kwargs: Any) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, **kwargs)


def _empty_panel(axis, message: str) -> None:
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="0.45",
    )


def _load_runs(metric_paths: Sequence[str | Path]) -> list[tuple[Path, list[dict[str, Any]]]]:
    if not metric_paths:
        raise ValueError("at least one metrics JSONL path is required")
    runs: list[tuple[Path, list[dict[str, Any]]]] = []
    for raw_path in metric_paths:
        path = Path(raw_path)
        records = deduplicate_records(load_training_history(path))
        if not records:
            raise ValueError(f"training history contains no complete records: {path}")
        runs.append((path, records))
    return runs


def _run_labels(
    runs: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]],
) -> list[str]:
    base = [str(records[0]["run_id"]) for _, records in runs]
    counts = {name: base.count(name) for name in set(base)}
    labels: list[str] = []
    for name, (path, records) in zip(base, runs):
        architecture = records[0]["architecture"]
        if counts[name] > 1:
            labels.append(f"{name} / {path.stem} ({architecture})")
        else:
            labels.append(f"{name} ({architecture})")
    return labels


def _atomic_save_png(figure, output_path: str | Path, *, dpi: int = 160) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.stem}.{os.getpid()}.{uuid4().hex}.tmp.png"
    )
    try:
        figure.savefig(temporary, format="png", dpi=dpi, bbox_inches="tight")
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return output


def plot_training_curves(
    metric_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 160,
    smoothing_window: int = 1,
) -> Path:
    """Create an atomic 2x3 PNG comparison for Dense and/or MoE runs."""
    if (
        isinstance(smoothing_window, bool)
        or not isinstance(smoothing_window, int)
        or smoothing_window < 1
    ):
        raise ValueError("smoothing_window must be a positive integer")
    runs = _load_runs(metric_paths)
    labels = _run_labels(runs)
    plt = _load_pyplot()
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    ce_axis, ppl_axis, lr_axis, grad_axis, throughput_axis, moe_axis = axes.flat
    colours = plt.get_cmap("tab10")

    any_ce = False
    any_ppl = False
    any_lr = False
    any_grad = False
    any_throughput = False
    any_moe = False
    any_dpo_loss = False
    any_dpo_margin = False
    any_dpo_accuracy = False
    any_grpo_loss = False
    any_grpo_reward = False
    any_grpo_success = False
    expert_counts: set[int] = set()
    load_axis = moe_axis.twinx()

    for run_index, ((_, records), label) in enumerate(zip(runs, labels)):
        colour = colours(run_index % 10)
        steps, train_ce = _series(records, "train_ce")
        _, val_ce = _series(records, "val_ce")
        _, total_loss = _series(records, "train_total_loss")
        train_ce = _smooth_finite_segments(train_ce, smoothing_window)
        total_loss = _smooth_finite_segments(total_loss, smoothing_window)
        smoothing_suffix = (
            f" (smooth {smoothing_window})" if smoothing_window > 1 else ""
        )
        if _has_finite(train_ce):
            ce_axis.plot(
                steps,
                train_ce,
                color=colour,
                label=f"{label} train CE{smoothing_suffix}",
            )
            any_ce = True
        if _has_finite(val_ce):
            ce_axis.plot(
                steps,
                val_ce,
                color=colour,
                linestyle="--",
                marker="o",
                markersize=2.5,
                label=f"{label} val CE",
            )
            any_ce = True
        if _has_finite(total_loss) and total_loss != train_ce:
            ce_axis.plot(
                steps,
                total_loss,
                color=colour,
                linestyle=":",
                alpha=0.8,
                label=f"{label} total loss{smoothing_suffix}",
            )
            any_ce = True

        # DPO has no token cross-entropy or perplexity objective.  Reuse the
        # first two panels for its directly meaningful optimization signals.
        _, train_dpo_loss = _series(records, "train_dpo_loss")
        _, val_dpo_loss = _series(records, "val_dpo_loss")
        train_dpo_loss = _smooth_finite_segments(train_dpo_loss, smoothing_window)
        if _has_finite(train_dpo_loss):
            ce_axis.plot(
                steps, train_dpo_loss, color=colour,
                label=f"{label} train DPO loss{smoothing_suffix}",
            )
            any_dpo_loss = True
        if _has_finite(val_dpo_loss):
            ce_axis.plot(
                steps, val_dpo_loss, color=colour, linestyle="--", marker="o", markersize=2.5,
                label=f"{label} val DPO loss",
            )
            any_dpo_loss = True

        _, val_ppl = _series(records, "val_ppl", positive=True)
        if _has_finite(val_ppl):
            ppl_axis.plot(
                steps,
                val_ppl,
                color=colour,
                marker="o",
                markersize=2.5,
                label=label,
            )
            any_ppl = True

        _, val_margin = _series(records, "val_margin")
        _, val_accuracy = _series(records, "val_preference_accuracy")
        if _has_finite(val_margin):
            ppl_axis.plot(
                steps, val_margin, color=colour, marker="o", markersize=2.5,
                label=f"{label} val margin",
            )
            any_dpo_margin = True
        if _has_finite(val_accuracy):
            ppl_axis.plot(
                steps, val_accuracy, color=colour, linestyle="--", marker="x", markersize=3,
                label=f"{label} val preference accuracy",
            )
            any_dpo_accuracy = True

        # GRPO is on-policy: reward and verifier success are more meaningful
        # than token CE/PPL.  Reuse the first two panels while retaining the
        # shared LR, gradient, throughput and MoE-router panels.
        _, train_grpo_loss = _series(records, "train_grpo_loss")
        train_grpo_loss = _smooth_finite_segments(train_grpo_loss, smoothing_window)
        if _has_finite(train_grpo_loss):
            ce_axis.plot(
                steps, train_grpo_loss, color=colour,
                label=f"{label} train GRPO loss{smoothing_suffix}",
            )
            any_grpo_loss = True
        _, train_reward = _series(records, "train_reward")
        _, val_reward = _series(records, "val_reward")
        _, val_success = _series(records, "val_success_rate")
        if _has_finite(train_reward):
            ppl_axis.plot(steps, train_reward, color=colour, label=f"{label} train reward")
            any_grpo_reward = True
        if _has_finite(val_reward):
            ppl_axis.plot(
                steps, val_reward, color=colour, linestyle="--", marker="o", markersize=2.5,
                label=f"{label} val reward",
            )
            any_grpo_reward = True
        if _has_finite(val_success):
            ppl_axis.plot(
                steps, val_success, color=colour, linestyle=":", marker="x", markersize=3,
                label=f"{label} val success rate",
            )
            any_grpo_success = True

        _, learning_rate = _series(records, "lr")
        _, grad_norm = _series(records, "grad_norm")
        _, tokens_per_s = _series(records, "tokens_per_s", positive=True)
        if _has_finite(learning_rate):
            lr_axis.plot(
                steps,
                learning_rate,
                color=colour,
                label=label,
            )
            any_lr = True
        if _has_finite(grad_norm):
            grad_axis.plot(
                steps,
                grad_norm,
                color=colour,
                label=label,
            )
            any_grad = True
        if _has_finite(tokens_per_s):
            throughput_axis.plot(
                steps,
                tokens_per_s,
                color=colour,
                label=label,
            )
            any_throughput = True

        if records[0]["architecture"] == "moe":
            for record in records:
                num_experts = record.get("num_experts")
                if (
                    isinstance(num_experts, int)
                    and not isinstance(num_experts, bool)
                    and num_experts > 0
                ):
                    expert_counts.add(num_experts)
            _, aux_weighted = _series(records, "router_aux_weighted")
            _, aux_raw = _series(records, "router_aux_raw")
            aux_values = aux_weighted if _has_finite(aux_weighted) else aux_raw
            aux_name = "weighted aux" if _has_finite(aux_weighted) else "raw aux"
            max_load = [_router_max_load(record) for record in records]
            if _has_finite(aux_values):
                moe_axis.plot(
                    steps,
                    aux_values,
                    color=colour,
                    label=f"{label} {aux_name}",
                )
                any_moe = True
            if _has_finite(max_load):
                load_axis.plot(
                    steps,
                    max_load,
                    color=colour,
                    linestyle="--",
                    label=f"{label} max load",
                )
                any_moe = True

    for num_experts in sorted(expert_counts):
        load_axis.axhline(
            1.0 / num_experts,
            color="0.35",
            linestyle=":",
            linewidth=1.0,
            label=f"uniform load 1/{num_experts}",
        )
        any_moe = True

    ce_axis.set_title("Cross Entropy / DPO Loss")
    ce_axis.set_xlabel("Optimizer step")
    ce_axis.set_ylabel("Loss")
    ce_axis.grid(alpha=0.25)
    if any_ce or any_dpo_loss or any_grpo_loss:
        _legend_if_present(ce_axis, fontsize=7)
    else:
        _empty_panel(ce_axis, "No CE or DPO-loss metrics")

    objective_second_panel = (any_dpo_margin or any_dpo_accuracy or any_grpo_reward or any_grpo_success) and not any_ppl
    if objective_second_panel and (any_grpo_reward or any_grpo_success):
        ppl_axis.set_title("GRPO Reward / Success Rate")
    else:
        ppl_axis.set_title("Preference Margin / Accuracy" if objective_second_panel else "Validation Perplexity")
    ppl_axis.set_xlabel("Optimizer step")
    ppl_axis.set_ylabel("Reward / success rate" if objective_second_panel and (any_grpo_reward or any_grpo_success) else ("Margin / accuracy" if objective_second_panel else "PPL (log scale)"))
    if not objective_second_panel:
        ppl_axis.set_yscale("log")
    ppl_axis.grid(alpha=0.25, which="both")
    if any_ppl or any_dpo_margin or any_dpo_accuracy or any_grpo_reward or any_grpo_success:
        _legend_if_present(ppl_axis, fontsize=7)
    else:
        _empty_panel(ppl_axis, "No PPL, DPO, or GRPO reward metrics")

    lr_axis.set_title("Learning Rate")
    lr_axis.set_xlabel("Optimizer step")
    lr_axis.set_ylabel("Learning rate")
    lr_axis.grid(alpha=0.25)
    if any_lr:
        _legend_if_present(lr_axis, fontsize=7)
    else:
        _empty_panel(lr_axis, "No learning-rate metrics")

    grad_axis.set_title("Gradient Norm")
    grad_axis.set_xlabel("Optimizer step")
    grad_axis.set_ylabel("Norm before clipping")
    grad_axis.grid(alpha=0.25)
    if any_grad:
        _legend_if_present(grad_axis, fontsize=7)
    else:
        _empty_panel(grad_axis, "No gradient-norm metrics")

    throughput_axis.set_title("Training Throughput")
    throughput_axis.set_xlabel("Optimizer step")
    throughput_axis.set_ylabel("Tokens / second")
    throughput_axis.grid(alpha=0.25)
    if any_throughput:
        _legend_if_present(throughput_axis, fontsize=7)
    else:
        _empty_panel(throughput_axis, "No throughput metrics")

    moe_axis.set_title("MoE Router Health")
    moe_axis.set_xlabel("Optimizer step")
    moe_axis.set_ylabel("Router auxiliary loss")
    load_axis.set_ylabel("Maximum expert load fraction")
    moe_axis.grid(alpha=0.25)
    if any_moe:
        left_handles, left_labels = moe_axis.get_legend_handles_labels()
        right_handles, right_labels = load_axis.get_legend_handles_labels()
        moe_axis.legend(
            left_handles + right_handles,
            left_labels + right_labels,
            fontsize=7,
        )
    else:
        _empty_panel(moe_axis, "N/A for Dense runs / no MoE metrics")

    if title is None:
        title = "Training Curves"
    figure.suptitle(title)
    try:
        return _atomic_save_png(figure, output_path, dpi=dpi)
    finally:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot JSONL v1 training histories as a 2x3 PNG dashboard."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        nargs="+",
        required=True,
        help="One or more JSONL history files (multiple paths compare runs)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="Trailing mean window for train CE/total loss; gaps reset the window",
    )
    args = parser.parse_args()

    output = plot_training_curves(
        args.metrics,
        args.output,
        title=args.title,
        dpi=args.dpi,
        smoothing_window=args.smoothing_window,
    )
    print(f"saved plot: {output}")


if __name__ == "__main__":
    main()
