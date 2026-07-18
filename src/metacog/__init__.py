"""Factorial metacognitive-signal analyses."""

from .directions import FactorialDirections, fit_factorial_directions
from .thresholding import ThresholdResult, apply_symmetric_confidence

__all__ = [
    "FactorialDirections",
    "ThresholdResult",
    "apply_symmetric_confidence",
    "fit_factorial_directions",
]
