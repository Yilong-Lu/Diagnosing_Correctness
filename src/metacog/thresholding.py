"""Self-judgement label construction and confidence filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    labels: np.ndarray
    keep: np.ndarray


def apply_symmetric_confidence(probabilities: np.ndarray, threshold: float) -> ThresholdResult:
    """Remove the open uncertainty band ``(1-threshold, threshold)``.

    Retained probabilities at or above ``threshold`` receive label one, and
    retained probabilities at or below ``1-threshold`` receive label zero.
    """

    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0.5, 1.0]")
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite values in [0, 1]")

    keep = (p <= 1.0 - threshold) | (p >= threshold)
    labels = (p >= threshold).astype(np.int8)
    return ThresholdResult(labels=labels, keep=keep)


def paired_confidence_mask(
    probabilities: np.ndarray,
    pair_ids: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Keep a pair only when every response in that pair is high confidence."""

    result = apply_symmetric_confidence(probabilities, threshold)
    groups = np.asarray(pair_ids)
    if groups.shape != result.keep.shape:
        raise ValueError("pair_ids and probabilities must have identical shapes")
    keep_by_group = {
        group: bool(np.all(result.keep[groups == group])) for group in np.unique(groups)
    }
    return np.asarray([keep_by_group[group] for group in groups], dtype=bool)


def quadrant_labels(objective_correctness: np.ndarray, self_judgement: np.ndarray) -> np.ndarray:
    """Return A/B/C/D labels for the OC by SJ factorial cells."""

    oc = np.asarray(objective_correctness, dtype=np.int8)
    sj = np.asarray(self_judgement, dtype=np.int8)
    if oc.shape != sj.shape or oc.ndim != 1:
        raise ValueError("objective_correctness and self_judgement must be aligned vectors")
    if not set(np.unique(oc)).issubset({0, 1}) or not set(np.unique(sj)).issubset({0, 1}):
        raise ValueError("factor labels must be binary")

    out = np.empty(oc.shape, dtype="U1")
    out[(oc == 1) & (sj == 1)] = "A"
    out[(oc == 1) & (sj == 0)] = "B"
    out[(oc == 0) & (sj == 1)] = "C"
    out[(oc == 0) & (sj == 0)] = "D"
    return out
