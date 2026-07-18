# Diagnosing Correctness Probes under Self-Judgement Confounding

This repository contains the code and processed data accompanying our study of
objective correctness and language-model self-judgement signals. It provides
the implementation of the factorial activation contrasts, question-grouped
validation, cross-domain
transfer analyses, data-construction checks, robustness controls, and manuscript
figure/table builders.

## Reproduction levels

The repository separates three levels of reproduction because full hidden-state
artifacts are several gigabytes and are not stored in Git.

1. **Paper reproduction (CPU):** rebuild manuscript tables and data figures from
   the released publication CSV files.
2. **Analysis reproduction (CPU):** recompute all statistics from activation
   artifacts that follow the documented schema.
3. **Activation reproduction (GPU):** extract final-answer-token hidden states
   from the released processed response records and public model checkpoints,
   score answer likelihoods, then run the same analysis pipeline.

The Math pass-at-eight pilot used to select the frozen Math question set is
provenance rather than the entry point of this artifact. Reproduction begins
from the frozen processed question/response records. No claim is made that
rerunning stochastic answer generation would recover byte-identical responses.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade 'pip>=23' 'setuptools>=68' wheel
pip install -e '.[dev]'
```

For hidden-state extraction, also install the GPU dependencies:

```bash
pip install -e '.[gpu,dev]'
```

## Quick checks

```bash
pytest
metacog-audit .
metacog-reproduce --input artifacts/publication --output outputs/paper
```

For a GPU example without scheduler- or institution-specific settings, run
`examples/extract_activations.sh`; the adjacent examples recreate answer
likelihood sidecars, literal Yes/No judgement logits, data-construction checks,
OOD nuisance controls, and cross-domain analyses.

See `docs/REPRODUCIBILITY.md` for the complete execution order and
`docs/RESULTS_MAP.md` for the mapping from paper claims to commands and files.
`docs/VALIDATION.md` records numerical regression checks against the frozen
manuscript outputs. The supplementary scoring-rule, strict-pair resampling, and
Movies multi-reference checks are exposed through `metacog-robustness`.

## Core definitions

Each response has objective correctness `OC` and a thresholded self-judgement
label `SJ`. With cells A=(1,1), B=(1,0), C=(0,1), and D=(0,0), the directions are

```text
W_mix   = mean(A) - mean(D)
W_meta  = 0.5 * ((mean(A) - mean(B)) + (mean(C) - mean(D)))
W_truth = 0.5 * ((mean(A) - mean(C)) + (mean(B) - mean(D)))
```

The primary confidence rule retains rows with `p(SJ=yes) <= 0.3` or
`p(SJ=yes) >= 0.7`; values in the open interval `(0.3, 0.7)` are removed.
Exp2A uses question-grouped five-fold cross-validation. Primary Exp2B inference
independently resamples source and target question clusters, refits directions,
and averages layer-wise AUCs over normalized depth `[0.40, 0.80]`.

## Data and path safety

Model paths are supplied at runtime and are never stored as absolute paths.
Generated metadata contains relative inputs, public model identifiers, explicit
seeds, and file checksums. `data/README.md`, `data/manifest.csv`, and
`docs/THIRD_PARTY.md` record benchmark provenance, applicable upstream terms,
and the composition of the Math source mixture.
