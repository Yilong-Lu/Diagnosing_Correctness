"""Question-cluster resampling and confidence intervals."""

from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "::".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def resample_cluster_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample clusters with replacement while retaining every row in a cluster."""

    group_array = np.asarray(groups)
    clusters = [np.flatnonzero(group_array == group) for group in np.unique(group_array)]
    sampled = rng.integers(0, len(clusters), len(clusters))
    parts = [clusters[int(position)] for position in sampled]
    return np.concatenate(parts) if parts else np.asarray([], dtype=np.int64)


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - level
    return tuple(float(x) for x in np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0]))


def cluster_bootstrap(
    groups: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.full(repetitions, np.nan, dtype=np.float64)
    for repetition in range(repetitions):
        indices = resample_cluster_indices(groups, rng)
        values[repetition] = statistic(indices)
    return values


def normalized_layer_mask(layers: np.ndarray, start: float, end: float) -> np.ndarray:
    layer_array = np.asarray(layers, dtype=np.float64)
    if layer_array.ndim != 1 or len(layer_array) == 0:
        raise ValueError("layers must be a non-empty vector")
    maximum = float(layer_array.max())
    if maximum <= 0:
        depth = np.zeros_like(layer_array)
    else:
        depth = layer_array / maximum
    return (depth >= start) & (depth <= end)
