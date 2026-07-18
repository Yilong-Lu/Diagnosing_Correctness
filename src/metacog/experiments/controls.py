"""OC-only, question-FE, nuisance, matching, and elicitation controls."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class PairedDifferenceData:
    differences: np.ndarray
    design: np.ndarray
    question_groups: np.ndarray
    pattern_counts: dict[str, int]
    rank: int

    @property
    def n_pairs(self) -> int:
        return int(len(self.design))


def paired_difference_design(
    activations: np.ndarray,
    objective_correctness: np.ndarray,
    self_judgement: np.ndarray,
    question_groups: np.ndarray,
) -> PairedDifferenceData:
    """Build correct-minus-incorrect outcomes for adjacent strict pairs."""

    x = np.asarray(activations, dtype=np.float32)
    oc = np.asarray(objective_correctness, dtype=np.int8)
    sj = np.asarray(self_judgement, dtype=np.int8)
    groups = np.asarray(question_groups)
    if x.ndim != 2 or len(x) != len(oc) or len(x) != len(sj) or len(x) != len(groups):
        raise ValueError("activations, labels, and question groups must have aligned rows")
    if len(x) == 0 or len(x) % 2:
        raise ValueError("strict-pair analysis requires a nonempty even number of rows")
    if not np.all(oc[0::2] == 1) or not np.all(oc[1::2] == 0):
        raise ValueError("strict pairs must place the correct response first")
    if not np.all(groups[0::2] == groups[1::2]):
        raise ValueError("strict-pair rows must share a question identifier")

    correct_sj = 2.0 * sj[0::2].astype(np.float32) - 1.0
    incorrect_sj = 2.0 * sj[1::2].astype(np.float32) - 1.0
    design = np.column_stack([
        np.full(len(correct_sj), 2.0, dtype=np.float32),
        correct_sj - incorrect_sj,
        correct_sj + incorrect_sj,
    ]).astype(np.float32)
    correct_cell = np.where(sj[0::2] == 1, "A", "B")
    incorrect_cell = np.where(sj[1::2] == 1, "C", "D")
    patterns = np.char.add(correct_cell, incorrect_cell)
    return PairedDifferenceData(
        differences=(x[0::2] - x[1::2]).astype(np.float32, copy=False),
        design=design,
        question_groups=groups[0::2],
        pattern_counts={
            pattern: int(np.sum(patterns == pattern))
            for pattern in ("AC", "AD", "BC", "BD")
        },
        rank=int(np.linalg.matrix_rank(design)),
    )


def weighted_paired_fe_directions(
    differences: np.ndarray,
    design: np.ndarray,
    pair_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate OC, SJ, and interaction vectors for weighted source pairs."""

    outcomes = np.asarray(differences, dtype=np.float32)
    predictors = np.asarray(design, dtype=np.float32)
    weights = np.asarray(pair_weights, dtype=np.float32)
    if weights.ndim == 1:
        weights = weights[None, :]
    if outcomes.ndim != 2 or predictors.shape != (len(outcomes), 3):
        raise ValueError("expected pair outcomes and a three-column aligned design")
    if weights.ndim != 2 or weights.shape[1] != len(outcomes) or np.any(weights < 0):
        raise ValueError("pair weights must be a nonnegative matrix aligned with pairs")

    coefficients = np.full(
        (len(weights), 3, outcomes.shape[1]), np.nan, dtype=np.float32
    )
    ranks = np.zeros(len(weights), dtype=np.int8)
    for repetition, repetition_weights in enumerate(weights):
        gram = predictors.T @ (predictors * repetition_weights[:, None])
        ranks[repetition] = np.linalg.matrix_rank(gram)
        if ranks[repetition] < 3:
            continue
        cross_product = predictors.T @ (outcomes * repetition_weights[:, None])
        coefficients[repetition] = np.linalg.solve(gram, cross_product).astype(np.float32)
    valid = ranks == 3
    return (
        coefficients[:, 1, :],
        coefficients[:, 0, :],
        coefficients[:, 2, :],
        valid,
        ranks,
    )


def source_question_fe_control(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    repetitions: int = 1000,
    seed: int = 20260715,
    window_start: float = 0.40,
    window_end: float = 0.80,
    batch_size: int = 20,
) -> dict[str, float]:
    """Refit source paired-FE directions in a joint source-target bootstrap."""

    if not np.array_equal(source.layers, target.layers):
        raise ValueError("source and target layers must match")
    positions = np.flatnonzero(
        normalized_layer_mask(source.layers, window_start, window_end)
    )
    if len(positions) == 0:
        raise ValueError("the requested layer window selects no layers")
    template = paired_difference_design(
        source.activations[:, positions[0], :],
        source.objective_correctness,
        source.self_judgement,
        source.question_ids,
    )
    if template.rank < 3:
        raise ValueError("source pair design is rank deficient")

    source_rng = np.random.default_rng(
        stable_seed(seed, source.model, source.domain, target.domain, "source_question_fe")
    )
    source_weights = np.empty((repetitions, template.n_pairs), dtype=np.float32)
    for repetition in range(repetitions):
        sampled_rows = resample_cluster_indices(source.question_ids, source_rng)
        row_weights = np.bincount(sampled_rows, minlength=source.n_samples)
        source_weights[repetition] = row_weights[0::2]

    target_mask = conflict_mask(target.objective_correctness, target.self_judgement)
    target_rows = np.flatnonzero(target_mask)
    target_groups = target.question_ids[target_mask]
    target_rng = np.random.default_rng(
        stable_seed(seed, source.model, source.domain, target.domain, "target_question_fe")
    )
    target_indices = [
        resample_cluster_indices(target_groups, target_rng) for _ in range(repetitions)
    ]
    target_sj = target.self_judgement[target_mask]
    target_oc = target.objective_correctness[target_mask]
    target_valid = np.asarray([
        len(np.unique(target_sj[indices])) == 2 for indices in target_indices
    ])

    source_valid = np.ones(repetitions, dtype=bool)
    meta_sum = np.zeros(repetitions, dtype=np.float64)
    truth_sum = np.zeros(repetitions, dtype=np.float64)
    layer_count = np.zeros(repetitions, dtype=np.int16)
    point_meta = []
    point_truth = []
    unit_weights = np.ones((1, template.n_pairs), dtype=np.float32)
    for position in positions:
        paired = paired_difference_design(
            source.activations[:, position, :],
            source.objective_correctness,
            source.self_judgement,
            source.question_ids,
        )
        target_x = target.activations[target_rows, position, :].astype(np.float32, copy=False)
        meta, truth, _, valid, _ = weighted_paired_fe_directions(
            paired.differences, paired.design, unit_weights
        )
        if not valid[0]:
            raise ValueError("source pair design is rank deficient")
        point_meta.append(binary_auc(target_sj, target_x @ meta[0]))
        point_truth.append(binary_auc(target_oc, target_x @ truth[0]))

        for begin in range(0, repetitions, batch_size):
            end = min(begin + batch_size, repetitions)
            meta, truth, _, valid, _ = weighted_paired_fe_directions(
                paired.differences, paired.design, source_weights[begin:end]
            )
            source_valid[begin:end] &= valid
            meta_scores = target_x @ meta.T
            truth_scores = target_x @ truth.T
            for local, repetition in enumerate(range(begin, end)):
                if not valid[local] or not target_valid[repetition]:
                    continue
                indices = target_indices[repetition]
                meta_auc = binary_auc(target_sj[indices], meta_scores[indices, local])
                truth_auc = binary_auc(target_oc[indices], truth_scores[indices, local])
                if np.isfinite(meta_auc) and np.isfinite(truth_auc):
                    meta_sum[repetition] += meta_auc
                    truth_sum[repetition] += truth_auc
                    layer_count[repetition] += 1

    complete = source_valid & target_valid & (layer_count == len(positions))
    bootstrap = np.full((repetitions, 3), np.nan, dtype=np.float64)
    bootstrap[complete, 0] = meta_sum[complete] / len(positions)
    bootstrap[complete, 1] = truth_sum[complete] / len(positions)
    bootstrap[complete, 2] = bootstrap[complete, 0] - bootstrap[complete, 1]
    observed = np.asarray([
        np.mean(point_meta), np.mean(point_truth), np.mean(point_meta) - np.mean(point_truth)
    ])
    result: dict[str, float] = {
        "window_start": float(window_start),
        "window_end": float(window_end),
        "n_layers": int(len(positions)),
        "source_design_rank": int(template.rank),
        "n_boot_requested": int(repetitions),
        "n_boot_valid": int(complete.sum()),
        "seed": int(seed),
    }
    result.update({f"source_pair_{key}": value for key, value in template.pattern_counts.items()})
    for column, name in enumerate(("meta_to_sj_auc", "truth_to_oc_auc", "delta_cb")):
        low, high = percentile_interval(bootstrap[:, column])
        result[name] = float(observed[column])
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high
    return result


def oc_only_conflict_auc(source: ActivationBundle, target: ActivationBundle) -> list[dict[str, float]]:
    """Evaluate the standard unwhitened OC class-mean contrast on target B/C rows."""

    if not np.array_equal(source.layers, target.layers):
        raise ValueError("source and target layers must match")
    mask = conflict_mask(target.objective_correctness, target.self_judgement)
    rows: list[dict[str, float]] = []
    for position, layer in enumerate(source.layers):
        directions = fit_factorial_directions(
            source.activations[:, position, :],
            source.objective_correctness,
            source.self_judgement,
        )
        scores = project(
            target.activations[:, position, :], directions.oc_only, directions.center
        )
        rows.append(
            {
                "layer": int(layer),
                "oc_only_c_above_b_auc": binary_auc(
                    target.self_judgement[mask], scores[mask]
                ),
                "n_conflict": int(mask.sum()),
            }
        )
    return rows


def oc_only_window_control(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    repetitions: int = 1000,
    seed: int = 20260707,
    window_start: float = 0.40,
    window_end: float = 0.80,
) -> dict[str, float]:
    """Question-bootstrap the fixed-window C-over-B AUC of the OC-only direction."""

    positions = np.flatnonzero(normalized_layer_mask(source.layers, window_start, window_end))
    mask = conflict_mask(target.objective_correctness, target.self_judgement)
    labels = target.self_judgement[mask]
    groups = target.question_ids[mask]
    matrices = component_score_matrix(source, target, component="oc_only", positions=positions)[:, mask]

    def evaluate(indices: np.ndarray) -> float:
        return float(np.nanmean([binary_auc(labels[indices], layer[indices]) for layer in matrices]))

    point = evaluate(np.arange(mask.sum()))
    rng = np.random.default_rng(seed)
    values = np.asarray(
        [evaluate(resample_cluster_indices(groups, rng)) for _ in range(repetitions)],
        dtype=np.float64,
    )
    low, high = percentile_interval(values)
    return {
        "auc_c_above_b": point,
        "auc_ci_low": low,
        "auc_ci_high": high,
        "p_le_0_5": float(np.mean(values <= 0.5)),
        "n_boot_valid": int(np.isfinite(values).sum()),
        "n_conflict": int(mask.sum()),
        "n_questions": int(len(np.unique(groups))),
        "n_layers": int(len(positions)),
    }


def residualize(scores: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    """OLS residualization with an intercept, used inside bootstrap controls."""

    y = np.asarray(scores, dtype=np.float64)
    z = np.asarray(nuisance, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    if y.ndim != 1 or len(y) != len(z):
        raise ValueError("scores and nuisance variables must have aligned rows")
    design = np.column_stack([np.ones(len(z)), z])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficients


def unit_vector(direction: np.ndarray) -> np.ndarray:
    vector = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if np.isfinite(norm) and norm > 0 else np.zeros_like(vector)


def zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    standard_deviation = float(np.nanstd(array, ddof=1))
    centered = array - float(np.nanmean(array))
    return centered / standard_deviation if standard_deviation > 0 else centered


def standardized_window_scores(score_matrix: np.ndarray) -> np.ndarray:
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0:
        raise ValueError("score_matrix must have shape (layers, rows)")
    return np.mean(np.vstack([zscore(layer) for layer in scores]), axis=0)


def component_score_matrix(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    component: str,
    positions: np.ndarray,
) -> np.ndarray:
    matrices = []
    for position in positions:
        fitted = fit_factorial_directions(
            source.activations[:, position, :],
            source.objective_correctness,
            source.self_judgement,
        )
        direction = unit_vector(getattr(fitted, component))
        matrices.append(
            project(target.activations[:, position, :], direction, fitted.center)
        )
    return np.asarray(matrices, dtype=np.float64)


def signed_design(objective_correctness: np.ndarray, self_judgement: np.ndarray) -> np.ndarray:
    oc = 2.0 * np.asarray(objective_correctness, dtype=np.float64) - 1.0
    sj = 2.0 * np.asarray(self_judgement, dtype=np.float64) - 1.0
    return np.column_stack([oc, sj, oc * sj])


def demean_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    if array.ndim == 1:
        sums = np.bincount(inverse, weights=array)
        return array - (sums / counts)[inverse]
    columns = []
    for column in range(array.shape[1]):
        sums = np.bincount(inverse, weights=array[:, column])
        columns.append(array[:, column] - (sums / counts)[inverse])
    return np.column_stack(columns)


def ols_question_fixed_effects(
    outcome: np.ndarray,
    predictors: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    y = demean_by_group(outcome, groups)
    x = demean_by_group(predictors, groups)
    return np.linalg.pinv(x.T @ x) @ x.T @ y


def question_fe_control(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    repetitions: int = 1000,
    seed: int = 20260712,
    window_start: float = 0.40,
    window_end: float = 0.80,
) -> dict[str, float]:
    """Fit the all-cell within-question OC, SJ, and interaction regression."""

    positions = np.flatnonzero(
        normalized_layer_mask(source.layers, window_start, window_end)
    )
    meta_matrix = component_score_matrix(source, target, component="meta", positions=positions)
    truth_matrix = component_score_matrix(source, target, component="truth", positions=positions)
    design = signed_design(target.objective_correctness, target.self_judgement)
    meta_beta = ols_question_fixed_effects(
        standardized_window_scores(meta_matrix), design, target.question_ids
    )
    truth_beta = ols_question_fixed_effects(
        standardized_window_scores(truth_matrix), design, target.question_ids
    )
    estimate = float(meta_beta[1] - truth_beta[0])

    unique_groups = np.unique(target.question_ids)
    group_rows = [np.flatnonzero(target.question_ids == group) for group in unique_groups]
    rng = np.random.default_rng(seed)
    coefficient_values = np.full((repetitions, 6), np.nan, dtype=np.float64)
    for _ in range(repetitions):
        repetition = _
        chosen = rng.integers(0, len(group_rows), len(group_rows))
        indices = np.concatenate([group_rows[position] for position in chosen])
        boot_groups = np.concatenate(
            [np.full(len(group_rows[position]), draw) for draw, position in enumerate(chosen)]
        )
        boot_design = design[indices]
        boot_meta = ols_question_fixed_effects(
            standardized_window_scores(meta_matrix[:, indices]), boot_design, boot_groups
        )
        boot_truth = ols_question_fixed_effects(
            standardized_window_scores(truth_matrix[:, indices]), boot_design, boot_groups
        )
        coefficient_values[repetition] = np.concatenate([boot_meta, boot_truth])
    delta_values = coefficient_values[:, 1] - coefficient_values[:, 3]
    low, high = percentile_interval(delta_values)
    names = (
        "meta_oc",
        "meta_sj",
        "meta_interaction",
        "truth_oc",
        "truth_sj",
        "truth_interaction",
    )
    point_values = np.concatenate([meta_beta, truth_beta])
    coefficients = {}
    for column, name in enumerate(names):
        coefficient_low, coefficient_high = percentile_interval(coefficient_values[:, column])
        coefficients[name] = {
            "estimate": float(point_values[column]),
            "ci_low": coefficient_low,
            "ci_high": coefficient_high,
        }
    specificity = {}
    for name, estimate_value, bootstrap_values in (
        (
            "meta_sj_minus_oc",
            meta_beta[1] - meta_beta[0],
            coefficient_values[:, 1] - coefficient_values[:, 0],
        ),
        (
            "truth_oc_minus_sj",
            truth_beta[0] - truth_beta[1],
            coefficient_values[:, 3] - coefficient_values[:, 4],
        ),
    ):
        specificity_low, specificity_high = percentile_interval(bootstrap_values)
        specificity[name] = {
            "estimate": float(estimate_value),
            "ci_low": specificity_low,
            "ci_high": specificity_high,
            "p_le_zero": float(np.mean(bootstrap_values <= 0.0)),
            "p_ge_zero": float(np.mean(bootstrap_values >= 0.0)),
        }
    return {
        "beta_sj_meta": float(meta_beta[1]),
        "beta_oc_truth": float(truth_beta[0]),
        "delta_fe": estimate,
        "delta_fe_ci_low": low,
        "delta_fe_ci_high": high,
        "delta_fe_p_le_zero": float(np.mean(delta_values <= 0.0)),
        "coefficients": coefficients,
        "specificity": specificity,
        "n_boot_valid": int(np.all(np.isfinite(coefficient_values), axis=1).sum()),
        "n_rows": target.n_samples,
        "n_questions": int(len(unique_groups)),
        "n_layers": int(len(positions)),
    }


def nuisance_window_control(
    source: ActivationBundle,
    target: ActivationBundle,
    nuisance: np.ndarray,
    *,
    repetitions: int = 1000,
    seed: int = 20260712,
    window_start: float = 0.40,
    window_end: float = 0.80,
) -> dict[str, float]:
    """Residualize each layer and refit nuisance regressions in every bootstrap."""

    positions = np.flatnonzero(
        normalized_layer_mask(source.layers, window_start, window_end)
    )
    mask = conflict_mask(target.objective_correctness, target.self_judgement)
    controls = np.asarray(nuisance, dtype=np.float64)[mask]
    meta_matrix = component_score_matrix(source, target, component="meta", positions=positions)[:, mask]
    truth_matrix = component_score_matrix(source, target, component="truth", positions=positions)[:, mask]
    oc = target.objective_correctness[mask]
    sj = target.self_judgement[mask]

    def evaluate(indices: np.ndarray) -> tuple[float, float, float]:
        meta_aucs = []
        truth_aucs = []
        for meta_scores, truth_scores in zip(meta_matrix[:, indices], truth_matrix[:, indices]):
            meta_aucs.append(binary_auc(sj[indices], residualize(meta_scores, controls[indices])))
            truth_aucs.append(binary_auc(oc[indices], residualize(truth_scores, controls[indices])))
        meta_auc = float(np.mean(meta_aucs))
        truth_auc = float(np.mean(truth_aucs))
        return meta_auc, truth_auc, meta_auc - truth_auc

    point = evaluate(np.arange(mask.sum()))
    rng = np.random.default_rng(seed)
    values = np.full((repetitions, 3), np.nan)
    groups = target.question_ids[mask]
    for repetition in range(repetitions):
        indices = resample_cluster_indices(groups, rng)
        values[repetition] = evaluate(indices)
    low, high = percentile_interval(values[:, 2])
    return {
        "resid_meta_to_sj_auc": point[0],
        "resid_truth_to_oc_auc": point[1],
        "resid_delta_cb": point[2],
        "resid_delta_cb_ci_low": low,
        "resid_delta_cb_ci_high": high,
        "resid_delta_cb_p_le_zero": float(np.mean(values[:, 2] <= 0.0)),
        "n_boot_valid": int(np.all(np.isfinite(values), axis=1).sum()),
    }


def standardized_mean_difference(labels: np.ndarray, values: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int8)
    x = np.asarray(values, dtype=np.float64)
    one = x[y == 1]
    zero = x[y == 0]
    if len(one) < 2 or len(zero) < 2:
        return float("nan")
    pooled = ((len(one) - 1) * one.var(ddof=1) + (len(zero) - 1) * zero.var(ddof=1))
    pooled /= len(one) + len(zero) - 2
    return float((one.mean() - zero.mean()) / np.sqrt(pooled)) if pooled > 0 else 0.0


def _evenly_spaced_indices(indices: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    ordered = indices[np.argsort(values[indices], kind="mergesort")]
    if len(ordered) == count:
        return ordered
    positions = np.floor((np.arange(count) + 0.5) * len(ordered) / count).astype(int)
    return ordered[positions]


def _coarsened_match(labels: np.ndarray, values: np.ndarray, bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 2:
        zero = np.flatnonzero(labels == 0)
        one = np.flatnonzero(labels == 1)
        count = min(len(zero), len(one))
        selected = np.concatenate(
            [
                _evenly_spaced_indices(zero, values, count),
                _evenly_spaced_indices(one, values, count),
            ]
        )
        return np.asarray(sorted(selected.tolist()), dtype=int)
    bin_ids = np.digitize(values, edges[1:-1], right=True)
    selected = []
    for bin_id in np.unique(bin_ids):
        in_bin = np.flatnonzero(bin_ids == bin_id)
        zero = in_bin[labels[in_bin] == 0]
        one = in_bin[labels[in_bin] == 1]
        count = min(len(zero), len(one))
        if count:
            selected.extend(_evenly_spaced_indices(zero, values, count))
            selected.extend(_evenly_spaced_indices(one, values, count))
    return np.asarray(sorted(selected), dtype=int)


def token_count_match(
    self_judgement: np.ndarray,
    token_count: np.ndarray,
    candidate_bins: tuple[int, ...] = (20, 10, 5, 2, 1),
) -> dict[str, object]:
    labels = np.asarray(self_judgement, dtype=np.int8)
    values = np.asarray(token_count, dtype=np.float64)
    candidates = [(bins, _coarsened_match(labels, values, bins)) for bins in candidate_bins]
    candidates = [(bins, indices) for bins, indices in candidates if len(indices)]
    if not candidates:
        return {"indices": np.asarray([], dtype=int), "n_bins": 0, "retention_rate": 0.0}
    bins, indices = min(
        candidates,
        key=lambda item: (
            abs(binary_auc(labels[item[1]], values[item[1]]) - 0.5),
            -len(item[1]),
        ),
    )
    return {
        "indices": indices,
        "n_bins": bins,
        "retention_rate": float(len(indices) / len(labels)),
        "smd_before": standardized_mean_difference(labels, values),
        "smd_after": standardized_mean_difference(labels[indices], values[indices]),
    }


def token_matched_window_control(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    repetitions: int = 1000,
    seed: int = 42,
    window_start: float = 0.40,
    window_end: float = 0.80,
    candidate_bins: tuple[int, ...] = (20, 10, 5, 2, 1),
) -> dict[str, float | int]:
    """Evaluate fixed-window component transfer after coarsened token-count matching."""

    conflict = conflict_mask(target.objective_correctness, target.self_judgement)
    token_counts = np.asarray([float(record["token_count"]) for record in target.records])[conflict]
    matched = token_count_match(
        target.self_judgement[conflict], token_counts, candidate_bins=candidate_bins
    )
    selected = np.asarray(matched["indices"], dtype=int)
    if len(selected) == 0:
        raise ValueError("token-count matching retained no conflict rows")
    positions = np.flatnonzero(normalized_layer_mask(source.layers, window_start, window_end))
    meta = component_score_matrix(source, target, component="meta", positions=positions)[:, conflict][:, selected]
    truth = component_score_matrix(source, target, component="truth", positions=positions)[:, conflict][:, selected]
    sj = target.self_judgement[conflict][selected]
    oc = target.objective_correctness[conflict][selected]
    groups = target.question_ids[conflict][selected]

    def evaluate(indices: np.ndarray) -> float:
        values = [
            binary_auc(sj[indices], meta_layer[indices])
            - binary_auc(oc[indices], truth_layer[indices])
            for meta_layer, truth_layer in zip(meta, truth)
        ]
        return float(np.nanmean(values))

    point = evaluate(np.arange(len(selected)))
    rng = np.random.default_rng(seed)
    values = np.asarray(
        [evaluate(resample_cluster_indices(groups, rng)) for _ in range(repetitions)],
        dtype=np.float64,
    )
    low, high = percentile_interval(values)
    return {
        "matched_delta_cb": point,
        "matched_delta_ci_low": low,
        "matched_delta_ci_high": high,
        "p_le_zero": float(np.mean(values <= 0.0)),
        "n_boot_valid": int(np.isfinite(values).sum()),
        "matched_n": int(len(selected)),
        "matched_questions": int(len(np.unique(groups))),
        "n_bins": int(matched["n_bins"]),
        "retention_rate": float(matched["retention_rate"]),
        "smd_before": float(matched["smd_before"]),
        "smd_after": float(matched["smd_after"]),
    }


def null_window_control(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    repetitions: int = 100,
    seed: int = 42,
    window_start: float = 0.40,
    window_end: float = 0.80,
) -> dict[str, float | int]:
    """Compare the observed window effect with source-label and random-direction nulls."""

    positions = np.flatnonzero(normalized_layer_mask(source.layers, window_start, window_end))
    conflict = conflict_mask(target.objective_correctness, target.self_judgement)
    target_oc = target.objective_correctness[conflict]
    target_sj = target.self_judgement[conflict]

    def delta(meta_scores: np.ndarray, truth_scores: np.ndarray) -> float:
        return binary_auc(target_sj, meta_scores[conflict]) - binary_auc(
            target_oc, truth_scores[conflict]
        )

    observed = []
    layer_inputs = []
    for position in positions:
        source_x = source.activations[:, position, :]
        target_x = target.activations[:, position, :]
        fitted = fit_factorial_directions(
            source_x, source.objective_correctness, source.self_judgement
        )
        observed.append(
            delta(
                project(target_x, fitted.meta, fitted.center),
                project(target_x, fitted.truth, fitted.center),
            )
        )
        layer_inputs.append((source_x, target_x, fitted))

    rng = np.random.default_rng(seed)
    shuffle_values = []
    random_values = []
    for _ in range(repetitions):
        shuffle_layers = []
        random_layers = []
        for source_x, target_x, fitted in layer_inputs:
            shuffled_sj = rng.permutation(source.self_judgement)
            shuffled_oc = rng.permutation(source.objective_correctness)
            meta_fit = fit_factorial_directions(
                source_x, source.objective_correctness, shuffled_sj
            )
            truth_fit = fit_factorial_directions(
                source_x, shuffled_oc, source.self_judgement
            )
            shuffle_layers.append(
                delta(
                    project(target_x, meta_fit.meta, meta_fit.center),
                    project(target_x, truth_fit.truth, truth_fit.center),
                )
            )
            target_centered = target_x - fitted.center
            random_meta = rng.normal(size=target_x.shape[1])
            random_truth = rng.normal(size=target_x.shape[1])
            random_meta *= np.linalg.norm(fitted.meta) / max(np.linalg.norm(random_meta), 1e-12)
            random_truth *= np.linalg.norm(fitted.truth) / max(np.linalg.norm(random_truth), 1e-12)
            random_layers.append(delta(target_centered @ random_meta, target_centered @ random_truth))
        shuffle_values.append(float(np.nanmean(shuffle_layers)))
        random_values.append(float(np.nanmean(random_layers)))

    shuffle_array = np.asarray(shuffle_values, dtype=np.float64)
    random_array = np.asarray(random_values, dtype=np.float64)
    shuffle_low, shuffle_high = percentile_interval(shuffle_array)
    random_low, random_high = percentile_interval(random_array)
    observed_mean = float(np.nanmean(observed))
    return {
        "observed_delta_cb": observed_mean,
        "label_shuffle_mean": float(np.nanmean(shuffle_array)),
        "label_shuffle_ci_low": shuffle_low,
        "label_shuffle_ci_high": shuffle_high,
        "label_shuffle_p_ge_observed": float((np.sum(shuffle_array >= observed_mean) + 1) / (len(shuffle_array) + 1)),
        "random_mean": float(np.nanmean(random_array)),
        "random_ci_low": random_low,
        "random_ci_high": random_high,
        "random_p_ge_observed": float((np.sum(random_array >= observed_mean) + 1) / (len(random_array) + 1)),
        "n_repetitions": int(repetitions),
        "n_layers": int(len(positions)),
    }


def counterbalanced_probability(
    xy_log_prob_x: float,
    xy_log_prob_y: float,
    yx_log_prob_x: float,
    yx_log_prob_y: float,
) -> float:
    """Average correctness-oriented XY and YX log odds, then apply sigmoid."""

    balanced = 0.5 * (
        (xy_log_prob_x - xy_log_prob_y) + (yx_log_prob_y - yx_log_prob_x)
    )
    return float(1.0 / (1.0 + np.exp(-balanced)))
