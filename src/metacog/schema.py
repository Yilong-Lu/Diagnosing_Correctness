"""Typed containers for processed response and activation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ActivationBundle:
    model: str
    domain: str
    activations: np.ndarray
    layers: np.ndarray
    sample_ids: np.ndarray
    question_ids: np.ndarray
    objective_correctness: np.ndarray
    self_judgement: np.ndarray
    p_self_judgement: np.ndarray
    records: Sequence[Mapping[str, Any]]
    source_dir: Path

    def __post_init__(self) -> None:
        n = self.activations.shape[0]
        if self.activations.ndim != 3:
            raise ValueError("activations must have shape (samples, layers, hidden_size)")
        if self.activations.shape[1] != len(self.layers):
            raise ValueError("layer metadata does not match activation shape")
        aligned = [
            self.sample_ids,
            self.question_ids,
            self.objective_correctness,
            self.self_judgement,
            self.p_self_judgement,
        ]
        if any(len(value) != n for value in aligned) or len(self.records) != n:
            raise ValueError("sample metadata is not aligned with activation rows")

    @property
    def n_samples(self) -> int:
        return int(self.activations.shape[0])

    @property
    def n_layers(self) -> int:
        return int(self.activations.shape[1])
