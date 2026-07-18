"""Exp2A: question-grouped in-domain validation of factorial directions."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from ..bootstrap import (
    cluster_bootstrap,
    normalized_layer_mask,
    percentile_interval,
    resample_cluster_indices,
    stable_seed,
)
from ..directions import fit_factorial_directions, project
from ..metrics import binary_auc, conflict_component_metrics, conflict_mask
from ..schema import ActivationBundle


def _out_of_fold_scores(
    bundle: ActivationBundle,
    position: int,
    *,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    unique_questions = np.unique(bundle.question_ids)
    splitter = GroupKFold(n_splits=min(n_splits, len(unique_questions)))
    meta_scores = np.full(bundle.n_samples, np.nan, dtype=np.float64)
    truth_scores = np.full(bundle.n_samples, np.nan, dtype=np.float64)
    fold_ids = np.full(bundle.n_samples, -1, dtype=np.int16)
    dummy = np.zeros(bundle.n_samples, dtype=np.int8)
    for fold, (train, test) in enumerate(
        splitter.split(dummy, groups=bundle.question_ids)
    ):
        directions = fit_factorial_directions(
            bundle.activations[train, position, :],
            bundle.objective_correctness[train],
            bundle.self_judgement[train],
        )
        x_test = bundle.activations[test, position, :]
        meta_scores[test] = project(x_test, directions.meta, directions.center)
        truth_scores[test] = project(x_test, directions.truth, directions.center)
        fold_ids[test] = fold
    if np.any(fold_ids < 0):
        raise RuntimeError("some samples did not receive out-of-fold scores")
    return meta_scores, truth_scores, int(splitter.n_splits)


def run_exp2a(
    bundle: ActivationBundle,
    *,
    n_splits: int = 5,
    bootstrap_repetitions: int = 0,
    seed: int = 20260712,
) -> list[dict[str, float]]:
    unique_questions = np.unique(bundle.question_ids)
    if len(unique_questions) < 2:
        raise ValueError("Exp2A requires at least two question clusters")
    rows: list[dict[str, float]] = []

    for position, layer in enumerate(bundle.layers):
        meta_scores, truth_scores, actual_splits = _out_of_fold_scores(
            bundle, position, n_splits=n_splits
        )
        metrics = conflict_component_metrics(
            bundle.objective_correctness,
            bundle.self_judgement,
            meta_scores,
            truth_scores,
        )
        row = {
            "layer": int(layer),
            **metrics,
            "n_questions": int(len(unique_questions)),
            "n_splits": actual_splits,
        }
        if bootstrap_repetitions > 0:
            mask = conflict_mask(bundle.objective_correctness, bundle.self_judgement)
            oc = bundle.objective_correctness[mask]
            sj = bundle.self_judgement[mask]
            meta = meta_scores[mask]
            truth = truth_scores[mask]
            values = cluster_bootstrap(
                bundle.question_ids[mask],
                lambda indices: binary_auc(sj[indices], meta[indices])
                - binary_auc(oc[indices], truth[indices]),
                repetitions=bootstrap_repetitions,
                seed=stable_seed(
                    seed,
                    bundle.model,
                    bundle.domain,
                    layer,
                    "grouped_exp2a_cluster",
                ),
            )
            low, high = percentile_interval(values)
            finite = values[np.isfinite(values)]
            row.update(
                {
                    "delta_cb_ci_low": low,
                    "delta_cb_ci_high": high,
                    "delta_cb_p_le_zero": float(np.mean(finite <= 0.0)),
                    "n_boot_valid": int(len(finite)),
                }
            )
        rows.append(row)
    return rows


def run_exp2a_window(
    bundle: ActivationBundle,
    *,
    n_splits: int = 5,
    repetitions: int = 1000,
    seed: int = 20260707,
    window_start: float = 0.40,
    window_end: float = 0.80,
) -> dict[str, float]:
    """Average grouped out-of-fold AUCs within a normalized layer window."""

    unique_questions = np.unique(bundle.question_ids)
    if len(unique_questions) < 2:
        raise ValueError("Exp2A requires at least two question clusters")
    selected = np.flatnonzero(
        normalized_layer_mask(bundle.layers, window_start, window_end)
    )
    if len(selected) == 0:
        raise ValueError("the requested layer window selects no layers")

    mask = conflict_mask(bundle.objective_correctness, bundle.self_judgement)
    oc = bundle.objective_correctness[mask]
    sj = bundle.self_judgement[mask]
    groups = bundle.question_ids[mask]
    meta_by_layer = []
    truth_by_layer = []
    actual_splits = 0
    for position in selected:
        meta_scores, truth_scores, actual_splits = _out_of_fold_scores(
            bundle, int(position), n_splits=n_splits
        )
        meta_by_layer.append(meta_scores[mask])
        truth_by_layer.append(truth_scores[mask])
    meta_matrix = np.asarray(meta_by_layer)
    truth_matrix = np.asarray(truth_by_layer)

    def statistic(indices: np.ndarray) -> np.ndarray:
        meta = np.nanmean([binary_auc(sj[indices], values[indices]) for values in meta_matrix])
        truth = np.nanmean([binary_auc(oc[indices], values[indices]) for values in truth_matrix])
        return np.asarray([meta, truth, meta - truth], dtype=np.float64)

    observed = statistic(np.arange(len(oc)))
    rng = np.random.default_rng(
        stable_seed(seed, bundle.model, bundle.domain, "grouped_exp2a_window")
    )
    bootstrap = np.full((repetitions, 3), np.nan, dtype=np.float64)
    for repetition in range(repetitions):
        bootstrap[repetition] = statistic(resample_cluster_indices(groups, rng))

    result: dict[str, float] = {
        "window_start": float(window_start),
        "window_end": float(window_end),
        "n_layers": int(len(selected)),
        "n_questions": int(len(unique_questions)),
        "n_splits": int(actual_splits),
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
