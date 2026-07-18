"""Factorial activation contrasts used throughout the experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorialDirections:
    mixed: np.ndarray
    meta: np.ndarray
    truth: np.ndarray
    oc_only: np.ndarray
    center: np.ndarray
    cell_counts: dict[str, int]


def _validate_inputs(
    activations: np.ndarray,
    objective_correctness: np.ndarray,
    self_judgement: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(activations, dtype=np.float32)
    oc = np.asarray(objective_correctness, dtype=np.int8)
    sj = np.asarray(self_judgement, dtype=np.int8)
    if x.ndim != 2 or oc.ndim != 1 or sj.ndim != 1:
        raise ValueError("expected a two-dimensional activation matrix and label vectors")
    if len(x) != len(oc) or len(x) != len(sj):
        raise ValueError("activations and labels must have equal row counts")
    if not set(np.unique(oc)).issubset({0, 1}) or not set(np.unique(sj)).issubset({0, 1}):
        raise ValueError("OC and SJ labels must be binary")
    return x, oc, sj


def fit_factorial_directions(
    activations: np.ndarray,
    objective_correctness: np.ndarray,
    self_judgement: np.ndarray,
) -> FactorialDirections:
    """Fit mixed, SJ-associated, OC-associated, and OC-only mean contrasts."""

    x, oc, sj = _validate_inputs(activations, objective_correctness, self_judgement)
    masks = {
        "A": (oc == 1) & (sj == 1),
        "B": (oc == 1) & (sj == 0),
        "C": (oc == 0) & (sj == 1),
        "D": (oc == 0) & (sj == 0),
    }
    missing = [cell for cell, mask in masks.items() if not np.any(mask)]
    if missing:
        raise ValueError(f"factorial direction requires all four cells; missing {missing}")
    center = x.mean(axis=0, dtype=np.float32)
    centered = x - center
    means = {
        cell: centered[mask].mean(axis=0, dtype=np.float32)
        for cell, mask in masks.items()
    }
    if not np.any(oc == 1) or not np.any(oc == 0):
        raise ValueError("OC-only direction requires both correctness classes")

    return FactorialDirections(
        mixed=means["A"] - means["D"],
        meta=0.5 * ((means["A"] - means["B"]) + (means["C"] - means["D"])),
        truth=0.5 * ((means["A"] - means["C"]) + (means["B"] - means["D"])),
        oc_only=centered[oc == 1].mean(axis=0, dtype=np.float32)
        - centered[oc == 0].mean(axis=0, dtype=np.float32),
        center=center,
        cell_counts={cell: int(mask.sum()) for cell, mask in masks.items()},
    )


def project(activations: np.ndarray, direction: np.ndarray, center: np.ndarray) -> np.ndarray:
    x = np.asarray(activations, dtype=np.float32)
    vector = np.asarray(direction, dtype=np.float32)
    origin = np.asarray(center, dtype=np.float32)
    if x.ndim != 2 or vector.ndim != 1 or origin.shape != vector.shape:
        raise ValueError("incompatible activation, direction, or centering shapes")
    return (x - origin) @ vector
