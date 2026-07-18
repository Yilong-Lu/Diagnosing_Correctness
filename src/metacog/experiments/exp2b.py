"""Exp2B cross-domain transfer and primary joint bootstrap inference."""

from __future__ import annotations

import numpy as np

from ..bootstrap import (
    normalized_layer_mask,
    percentile_interval,
    resample_cluster_indices,
    stable_seed,
)
from ..directions import fit_factorial_directions, project
from ..metrics import binary_auc, conflict_component_metrics, conflict_mask
from ..schema import ActivationBundle


def _check_layers(source: ActivationBundle, target: ActivationBundle) -> None:
    if not np.array_equal(source.layers, target.layers):
        raise ValueError("source and target bundles must contain identical layer indices")


def run_exp2b(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    bootstrap_repetitions: int = 0,
    seed: int = 20260712,
) -> list[dict[str, float]]:
    _check_layers(source, target)
    rows: list[dict[str, float]] = []
    for position, layer in enumerate(source.layers):
        directions = fit_factorial_directions(
            source.activations[:, position, :],
            source.objective_correctness,
            source.self_judgement,
        )
        target_x = target.activations[:, position, :]
        meta_scores = project(target_x, directions.meta, directions.center)
        truth_scores = project(target_x, directions.truth, directions.center)
        row = {
            "layer": int(layer),
            **conflict_component_metrics(
                target.objective_correctness,
                target.self_judgement,
                meta_scores,
                truth_scores,
            ),
        }
        if bootstrap_repetitions > 0:
            mask = conflict_mask(target.objective_correctness, target.self_judgement)
            oc = target.objective_correctness[mask]
            sj = target.self_judgement[mask]
            meta = meta_scores[mask]
            truth = truth_scores[mask]

            def statistic(indices: np.ndarray) -> np.ndarray:
                meta_auc = binary_auc(sj[indices], meta[indices])
                truth_auc = binary_auc(oc[indices], truth[indices])
                return np.asarray([meta_auc, truth_auc, meta_auc - truth_auc])

            rng = np.random.default_rng(
                stable_seed(
                    seed,
                    source.model,
                    layer,
                    source.domain,
                    target.domain,
                    "exp2b_question_cluster",
                )
            )
            values = np.full((bootstrap_repetitions, 3), np.nan, dtype=np.float64)
            groups = target.question_ids[mask]
            for repetition in range(bootstrap_repetitions):
                indices = resample_cluster_indices(groups, rng)
                values[repetition] = statistic(indices)
            for column, name, null in (
                (0, "meta_to_sj_auc", 0.5),
                (1, "truth_to_oc_auc", 0.5),
                (2, "delta_cb", 0.0),
            ):
                low, high = percentile_interval(values[:, column])
                finite = values[np.isfinite(values[:, column]), column]
                row[f"{name}_ci_low"] = low
                row[f"{name}_ci_high"] = high
                row[f"{name}_p_le_null"] = float(np.mean(finite <= null))
            row["n_boot_valid"] = int(np.all(np.isfinite(values), axis=1).sum())
        rows.append(row)
    return rows


def run_exp2b_window(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    repetitions: int = 1000,
    seed: int = 20260702,
    window_start: float = 0.40,
    window_end: float = 0.80,
) -> dict[str, float]:
    """Evaluate fixed source directions with target-question bootstrap inference."""

    _check_layers(source, target)
    selected = np.flatnonzero(
        normalized_layer_mask(source.layers, window_start, window_end)
    )
    if len(selected) == 0:
        raise ValueError("the requested layer window selects no layers")
    mask = conflict_mask(target.objective_correctness, target.self_judgement)
    oc = target.objective_correctness[mask]
    sj = target.self_judgement[mask]
    groups = target.question_ids[mask]
    meta_by_layer = []
    truth_by_layer = []
    for position in selected:
        directions = fit_factorial_directions(
            source.activations[:, position, :],
            source.objective_correctness,
            source.self_judgement,
        )
        target_x = target.activations[mask, position, :]
        meta_by_layer.append(project(target_x, directions.meta, directions.center))
        truth_by_layer.append(project(target_x, directions.truth, directions.center))
    meta_matrix = np.asarray(meta_by_layer)
    truth_matrix = np.asarray(truth_by_layer)

    def statistic(indices: np.ndarray) -> np.ndarray:
        meta = np.nanmean([binary_auc(sj[indices], values[indices]) for values in meta_matrix])
        truth = np.nanmean([binary_auc(oc[indices], values[indices]) for values in truth_matrix])
        return np.asarray([meta, truth, meta - truth], dtype=np.float64)

    observed = statistic(np.arange(len(oc)))
    rng = np.random.default_rng(
        stable_seed(
            seed,
            source.model,
            source.domain,
            target.domain,
            "fixed_source_window",
        )
    )
    bootstrap = np.full((repetitions, 3), np.nan, dtype=np.float64)
    for repetition in range(repetitions):
        bootstrap[repetition] = statistic(resample_cluster_indices(groups, rng))

    result: dict[str, float] = {
        "window_start": float(window_start),
        "window_end": float(window_end),
        "n_layers": int(len(selected)),
        "n_target_questions": int(len(np.unique(groups))),
        "n_boot_requested": int(repetitions),
        "n_boot_valid": int(np.all(np.isfinite(bootstrap), axis=1).sum()),
        "seed": int(seed),
    }
    for column, name, null in (
        (0, "meta_to_sj_auc", 0.5),
        (1, "truth_to_oc_auc", 0.5),
        (2, "delta_cb", 0.0),
    ):
        low, high = percentile_interval(bootstrap[:, column])
        finite = bootstrap[np.isfinite(bootstrap[:, column]), column]
        result[name] = float(observed[column])
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high
        result[f"{name}_p_le_null"] = float(np.mean(finite <= null))
    return result


def _window_statistic(
    source: ActivationBundle,
    target: ActivationBundle,
    source_weights: np.ndarray,
    target_indices: np.ndarray,
    selected_positions: np.ndarray,
) -> tuple[float, float, float]:
    meta_aucs = []
    truth_aucs = []
    for position in selected_positions:
        source_x = source.activations[:, position, :].astype(np.float32, copy=False)
        masks = (
            (source.objective_correctness == 1) & (source.self_judgement == 1),
            (source.objective_correctness == 1) & (source.self_judgement == 0),
            (source.objective_correctness == 0) & (source.self_judgement == 1),
            (source.objective_correctness == 0) & (source.self_judgement == 0),
        )
        denominators = [float(source_weights[mask].sum()) for mask in masks]
        if any(value <= 0 for value in denominators):
            return (float("nan"), float("nan"), float("nan"))
        means = [
            (source_weights[mask] @ source_x[mask]) / denominator
            for mask, denominator in zip(masks, denominators)
        ]
        a, b, c, d = means
        meta_direction = 0.5 * ((a - b) + (c - d))
        truth_direction = 0.5 * ((a - c) + (b - d))
        target_x = target.activations[target_indices, position, :].astype(np.float32, copy=False)
        metrics = conflict_component_metrics(
            target.objective_correctness[target_indices],
            target.self_judgement[target_indices],
            target_x @ meta_direction,
            target_x @ truth_direction,
        )
        meta_aucs.append(metrics["meta_to_sj_auc"])
        truth_aucs.append(metrics["truth_to_oc_auc"])
    meta = float(np.mean(meta_aucs))
    truth = float(np.mean(truth_aucs))
    return meta, truth, meta - truth


def run_joint_source_target_bootstrap(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    repetitions: int = 1000,
    seed: int = 20260712,
    window_start: float = 0.40,
    window_end: float = 0.80,
) -> dict[str, float]:
    """Independently resample source and target questions and refit directions."""

    _check_layers(source, target)
    selected = np.flatnonzero(
        normalized_layer_mask(source.layers, window_start, window_end)
    )
    if len(selected) == 0:
        raise ValueError("the requested layer window selects no layers")

    observed = _window_statistic(
        source,
        target,
        np.ones(source.n_samples, dtype=np.float32),
        np.arange(target.n_samples),
        selected,
    )
    source_rng = np.random.default_rng(
        stable_seed(seed, source.model, source.domain, target.domain, "source")
    )
    target_rng = np.random.default_rng(
        stable_seed(seed, source.model, source.domain, target.domain, "target")
    )
    bootstrap = np.full((repetitions, 3), np.nan, dtype=np.float64)
    target_conflict_indices = np.flatnonzero(
        conflict_mask(target.objective_correctness, target.self_judgement)
    )
    target_conflict_groups = target.question_ids[target_conflict_indices]
    for repetition in range(repetitions):
        source_indices = resample_cluster_indices(source.question_ids, source_rng)
        source_weights = np.bincount(
            source_indices, minlength=source.n_samples
        ).astype(np.float32)
        relative_target_indices = resample_cluster_indices(target_conflict_groups, target_rng)
        target_indices = target_conflict_indices[relative_target_indices]
        bootstrap[repetition] = _window_statistic(
            source, target, source_weights, target_indices, selected
        )

    names = ("meta_to_sj_auc", "truth_to_oc_auc", "delta_cb")
    result: dict[str, float] = {
        "window_start": float(window_start),
        "window_end": float(window_end),
        "n_layers": int(len(selected)),
        "n_boot_requested": int(repetitions),
        "n_boot_valid": int(np.all(np.isfinite(bootstrap), axis=1).sum()),
        "seed": int(seed),
    }
    for column, name in enumerate(names):
        low, high = percentile_interval(bootstrap[:, column])
        result[name] = float(observed[column])
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high
    return result
