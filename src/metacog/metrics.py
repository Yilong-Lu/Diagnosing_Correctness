"""Evaluation metrics for component and conflict-set analyses."""

from __future__ import annotations

import math

import numpy as np


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    if len(y) == 0 or len(np.unique(y)) != 2:
        return math.nan
    positive = y == 1
    n_positive = int(positive.sum())
    n_negative = int(len(y) - n_positive)
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    sorted_ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        sorted_ranks[start:end] = 0.5 * (start + 1 + end)
        start = end
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = sorted_ranks
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def cohens_d(group_one: np.ndarray, group_zero: np.ndarray) -> float:
    one = np.asarray(group_one, dtype=np.float64)
    zero = np.asarray(group_zero, dtype=np.float64)
    if len(one) < 2 or len(zero) < 2:
        return math.nan
    variance = ((len(one) - 1) * one.var(ddof=1) + (len(zero) - 1) * zero.var(ddof=1))
    variance /= len(one) + len(zero) - 2
    if variance <= 0:
        return math.nan
    return float((one.mean() - zero.mean()) / np.sqrt(variance))


def conflict_mask(objective_correctness: np.ndarray, self_judgement: np.ndarray) -> np.ndarray:
    return np.asarray(objective_correctness) != np.asarray(self_judgement)


def conflict_component_metrics(
    objective_correctness: np.ndarray,
    self_judgement: np.ndarray,
    meta_scores: np.ndarray,
    truth_scores: np.ndarray,
) -> dict[str, float]:
    mask = conflict_mask(objective_correctness, self_judgement)
    oc = np.asarray(objective_correctness, dtype=np.int8)[mask]
    sj = np.asarray(self_judgement, dtype=np.int8)[mask]
    meta_auc = binary_auc(sj, np.asarray(meta_scores)[mask])
    truth_auc = binary_auc(oc, np.asarray(truth_scores)[mask])
    return {
        "meta_to_sj_auc": meta_auc,
        "truth_to_oc_auc": truth_auc,
        "delta_cb": float(meta_auc - truth_auc),
        "n_conflict": int(mask.sum()),
    }
