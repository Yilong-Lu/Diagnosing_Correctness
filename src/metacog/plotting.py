"""Compact, publication-oriented plots from released source-data tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = ["qwen25_7b", "llama31_8b", "qwen25_14b", "olmo3_7b_it"]
MODEL_LABELS = {
    "qwen25_7b": "Qwen2.5-7B",
    "llama31_8b": "Llama-3.1-8B",
    "qwen25_14b": "Qwen2.5-14B",
    "olmo3_7b_it": "OLMo-3-7B",
}
META_COLOR = "#2166AC"
TRUTH_COLOR = "#D6604D"
MATH_COLOR = "#007C91"
MOVIES_COLOR = "#7B3294"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _panel_label(ax, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, weight="bold", fontsize=8)


def plot_exp1(table: pd.DataFrame, output: Path) -> None:
    configure_style()
    data = table[
        (table["experiment"] == "Exp1") & (table["source"] != table["target"])
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.35), sharey=True)
    for column, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        subset = data[data["model"] == model]
        for source, color, label in (
            ("math", MATH_COLOR, "Math to Movies"),
            ("movies", MOVIES_COLOR, "Movies to Math"),
        ):
            rows = subset[subset["source"] == source].sort_values("layer")
            ax.plot(rows["layer"], rows["auc_mj"], color=color, label=label)
            ax.fill_between(
                rows["layer"].to_numpy(),
                rows["auc_mj_ci_low"].to_numpy(),
                rows["auc_mj_ci_high"].to_numpy(),
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.7)
        ax.set_title(MODEL_LABELS[model], pad=5)
        ax.set_xlabel("Layer")
        ax.minorticks_on()
        _panel_label(ax, chr(ord("a") + column))
    axes[0].set_ylabel("AUC (C over B)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.80, wspace=0.18)
    _save(fig, output, "figure2_exp1")


def plot_exp2b(table: pd.DataFrame, output: Path) -> None:
    configure_style()
    data = table[table["experiment"] == "Exp2B"]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.75), sharex="col", sharey=True)
    directions = [("math", "Math to Movies"), ("movies", "Movies to Math")]
    for row_index, (source, row_label) in enumerate(directions):
        for column, model in enumerate(MODEL_ORDER):
            ax = axes[row_index, column]
            rows = data[(data["model"] == model) & (data["source"] == source)].sort_values("layer")
            for value, low, high, color, label in (
                ("cb_meta_mj_auc", "cb_meta_mj_auc_ci_low", "cb_meta_mj_auc_ci_high", META_COLOR, r"$W_{meta}$ to SJ"),
                ("cb_truth_gt_auc", "cb_truth_gt_auc_ci_low", "cb_truth_gt_auc_ci_high", TRUTH_COLOR, r"$W_{truth}$ to OC"),
            ):
                ax.plot(rows["layer"], rows[value], color=color, label=label)
                ax.fill_between(rows["layer"].to_numpy(), rows[low].to_numpy(), rows[high].to_numpy(), color=color, alpha=0.14, linewidth=0)
            ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.7)
            ax.minorticks_on()
            if row_index == 0:
                ax.set_title(MODEL_LABELS[model], pad=5)
            if column == 0:
                ax.set_ylabel(f"{row_label}\nAUC")
            if row_index == 1:
                ax.set_xlabel("Layer")
        _panel_label(axes[row_index, 0], chr(ord("a") + row_index))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.87, hspace=0.20, wspace=0.18)
    _save(fig, output, "figure3_exp2b")


def plot_ood(
    table: pd.DataFrame,
    output: Path,
    *,
    stem: str = "figure4_ood",
) -> None:
    configure_style()
    conditions = [
        ("mmlu", "math", "MMLU, Math source"),
        ("mmlu", "movies", "MMLU, Movies source"),
        ("truthfulqa_binary", "math", "TruthfulQA, Math source"),
        ("truthfulqa_binary", "movies", "TruthfulQA, Movies source"),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(7.0, 5.6), sharey=True)
    for row_index, (target, source, row_label) in enumerate(conditions):
        for column, model in enumerate(MODEL_ORDER):
            ax = axes[row_index, column]
            rows = table[
                (table["model"] == model)
                & (table["target"] == target)
                & (table["source"] == source)
            ].sort_values("layer")
            for value, low, high, color, label in (
                ("meta_mj_auc", "meta_mj_auc_ci_low", "meta_mj_auc_ci_high", META_COLOR, r"$W_{meta}$ to SJ"),
                ("truth_gt_auc", "truth_gt_auc_ci_low", "truth_gt_auc_ci_high", TRUTH_COLOR, r"$W_{truth}$ to OC"),
            ):
                ax.plot(rows["layer"], rows[value], color=color, label=label)
                ax.fill_between(rows["layer"].to_numpy(), rows[low].to_numpy(), rows[high].to_numpy(), color=color, alpha=0.14, linewidth=0)
            ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.7)
            ax.minorticks_on()
            if row_index == 0:
                ax.set_title(MODEL_LABELS[model], pad=5)
            if column == 0:
                ax.set_ylabel(f"{row_label}\nAUC")
            if row_index == 3:
                ax.set_xlabel("Layer")
        _panel_label(axes[row_index, 0], chr(ord("a") + row_index))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.subplots_adjust(top=0.92, hspace=0.25, wspace=0.18)
    _save(fig, output, stem)


def plot_exp2a(table: pd.DataFrame, output: Path) -> None:
    """Plot grouped out-of-fold in-domain component curves."""

    configure_style()
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.75), sharex="col", sharey=True)
    for row_index, domain in enumerate(("math", "movies")):
        for column, model in enumerate(MODEL_ORDER):
            ax = axes[row_index, column]
            rows = table[
                (table["model"] == model) & (table["domain"] == domain)
            ].sort_values("layer")
            for value, low, high, color, label in (
                (
                    "cb_meta_mj_auc",
                    "cb_meta_mj_auc_ci_low",
                    "cb_meta_mj_auc_ci_high",
                    META_COLOR,
                    r"$W_{meta}$ to SJ",
                ),
                (
                    "cb_truth_gt_auc",
                    "cb_truth_gt_auc_ci_low",
                    "cb_truth_gt_auc_ci_high",
                    TRUTH_COLOR,
                    r"$W_{truth}$ to OC",
                ),
            ):
                ax.plot(rows["layer"], rows[value], color=color, label=label)
                ax.fill_between(
                    rows["layer"].to_numpy(),
                    rows[low].to_numpy(),
                    rows[high].to_numpy(),
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
            ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.7)
            ax.minorticks_on()
            if row_index == 0:
                ax.set_title(MODEL_LABELS[model], pad=5)
            if column == 0:
                ax.set_ylabel(f"{domain.title()}\nAUC")
            if row_index == 1:
                ax.set_xlabel("Layer")
        _panel_label(axes[row_index, 0], chr(ord("a") + row_index))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.87, hspace=0.20, wspace=0.18)
    _save(fig, output, "figureS_exp2a_in_domain")


def plot_null_controls(table: pd.DataFrame, output: Path) -> None:
    """Show observed fixed-window effects against the two null intervals."""

    configure_style()
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.2), sharex=True, sharey=True)
    baselines = ("source_label_shuffle", "random_direction")
    baseline_labels = ("Label shuffle", "Random direction")
    for row_index, source in enumerate(("math", "movies")):
        for column, model in enumerate(MODEL_ORDER):
            ax = axes[row_index, column]
            rows = table[(table["model"] == model) & (table["source"] == source)]
            for y, (baseline, label) in enumerate(zip(baselines, baseline_labels)):
                item = rows[rows["baseline"] == baseline].iloc[0]
                ax.errorbar(
                    float(item["null_window_mean"]),
                    y,
                    xerr=[
                        [float(item["null_window_mean"] - item["null_window_ci_low"])],
                        [float(item["null_window_ci_high"] - item["null_window_mean"])],
                    ],
                    fmt="o",
                    color="#777777",
                    markersize=3,
                    capsize=2,
                )
            observed = float(rows["real_window_mean_delta"].iloc[0])
            ax.axvline(observed, color=META_COLOR, linewidth=1.2)
            ax.axvline(0, color="#BBBBBB", linestyle="--", linewidth=0.7)
            ax.set_yticks(range(len(baseline_labels)), baseline_labels)
            ax.minorticks_on()
            if row_index == 0:
                ax.set_title(MODEL_LABELS[model], pad=5)
            if column == 0:
                ax.set_ylabel(f"{source.title()} source")
            if row_index == 1:
                ax.set_xlabel(r"Null $\Delta_{CB}$")
        _panel_label(axes[row_index, 0], chr(ord("a") + row_index))
    fig.subplots_adjust(top=0.87, hspace=0.25, wspace=0.18)
    _save(fig, output, "figureS_window_null_controls")


def plot_threshold_distributions(data_root: Path, output: Path) -> None:
    """Plot pre-filter self-judgement probability distributions."""

    configure_style()
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.4), sharex=True, sharey="row")
    bins = np.linspace(0.0, 1.0, 41)
    for row_index, domain in enumerate(("math", "movies")):
        for column, model in enumerate(MODEL_ORDER):
            ax = axes[row_index, column]
            path = data_root / "processed" / "id" / model / domain / "all_pairs.jsonl"
            probabilities = [
                float(json.loads(line)["p_self_judgement"])
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ax.hist(probabilities, bins=bins, density=True, color=META_COLOR, alpha=0.72)
            for boundary in (0.2, 0.3, 0.4, 0.6, 0.7, 0.8):
                ax.axvline(boundary, color="#777777", linewidth=0.45, alpha=0.55)
            if row_index == 0:
                ax.set_title(MODEL_LABELS[model], pad=5)
            if column == 0:
                ax.set_ylabel(f"{domain.title()}\nDensity")
            if row_index == 1:
                ax.set_xlabel(r"$p_{judge}$")
        _panel_label(axes[row_index, 0], chr(ord("a") + row_index))
    fig.subplots_adjust(top=0.88, hspace=0.24, wspace=0.18)
    _save(fig, output, "figureS_self_judgement_distributions")
