"""Zero-target-fitting OOD transfer uses the same Exp2B estimator."""

from __future__ import annotations

from ..schema import ActivationBundle
from .exp2b import run_exp2b


def run_ood_transfer(
    source: ActivationBundle,
    target: ActivationBundle,
    *,
    bootstrap_repetitions: int = 0,
    seed: int = 20260712,
) -> list[dict[str, float]]:
    return run_exp2b(
        source,
        target,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
