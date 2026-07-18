"""Exp1: diagnose the ordering learned by the aligned A-minus-D contrast."""

from __future__ import annotations

import numpy as np

from ..bootstrap import cluster_bootstrap, percentile_interval, stable_seed
from ..directions import fit_factorial_directions, project
from ..metrics import binary_auc, cohens_d, conflict_mask
from ..schema import ActivationBundle


def run_exp1(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    bootstrap_repetitions: int = 0,
    seed: int = 20260712,
) -> list[dict[str, float]]:
    if not np.array_equal(source.layers, target.layers):
        raise ValueError("source and target bundles must contain the same layer indices")
    target_conflict = conflict_mask(target.objective_correctness, target.self_judgement)
    target_sj = target.self_judgement[target_conflict]
    rows: list[dict[str, float]] = []

    for position, layer in enumerate(source.layers):
        directions = fit_factorial_directions(
            source.activations[:, position, :],
            source.objective_correctness,
            source.self_judgement,
        )
        scores = project(
            target.activations[:, position, :], directions.mixed, directions.center
        )[target_conflict]
        row = {
            "layer": int(layer),
            "auc_c_over_b": binary_auc(target_sj, scores),
            "cohens_d_c_over_b": cohens_d(scores[target_sj == 1], scores[target_sj == 0]),
            "n_conflict": int(target_conflict.sum()),
        }
        if bootstrap_repetitions > 0:
            values = cluster_bootstrap(
                target.question_ids[target_conflict],
                lambda indices: binary_auc(target_sj[indices], scores[indices]),
                repetitions=bootstrap_repetitions,
                seed=stable_seed(
                    seed,
                    source.model,
                    layer,
                    source.domain,
                    target.domain,
                    "exp1_question_cluster",
                ),
            )
            low, high = percentile_interval(values)
            finite = values[np.isfinite(values)]
            row.update(
                {
                    "auc_c_over_b_ci_low": low,
                    "auc_c_over_b_ci_high": high,
                    "auc_c_over_b_p_le_chance": float(np.mean(finite <= 0.5)),
                    "n_boot_valid": int(len(finite)),
                }
            )
        rows.append(row)
    return rows
