# Reproducibility Guide

## Recommended order

1. Install the package and run `pytest`.
2. Run `metacog-audit .` to check the release source tree.
3. Run `metacog-reproduce` to rebuild publication tables and figures from the
   lightweight release artifacts.
4. To recompute statistics, place activation artifacts under `activations/` and
   run `metacog-analyze` with the desired experiment and configuration.
5. To recreate activations, run `metacog-extract` on processed response records
   using a public checkpoint or a local checkpoint path supplied on the command
   line.
6. Run `metacog-score-answer` on the same ordered records to recreate the
   answer-likelihood nuisance sidecar used by the main-domain and OOD controls.
7. Use `metacog-robustness` for the fixed-logit scoring-rule audit, the repeated
   strict-pair draw, and the Movies multi-reference filter in Supplementary
   Sec. S8.7.

The processed JSON Lines contract and normalization commands are documented in
`docs/DATA_SCHEMA.md`.
Portable shell examples are provided in `examples/`.

## Frozen inputs

The reproducible analysis begins from processed response records containing the
question, answer, objective-correctness label, and continuous self-judgement
probability. The primary strict sample is derived deterministically with the
symmetric confidence rule at `tau=0.7`.

The stochastic Math pilot, free-response generation, and realized draw of one
correct and one incorrect response per eligible question are not replayed. Their
outputs are frozen inputs because the historical pipeline did not record every
source of nondeterminism and because model-serving stacks can change generated
text even when a nominal seed is supplied.

The historical one-token self-judgement call retained up to four next-token
log-probability candidates. When literal `Yes` and `No` were both present, their
probabilities were normalized over the pair. A lone `Yes` used its stored
probability, a lone `No` used one minus its stored probability, and a row with
neither candidate received 0.5. The released processed records contain the resulting frozen
continuous probability; the primary symmetric threshold at `tau=0.7` excludes
the final-fallback value 0.5.

The scoring-rule audit uses `metacog-score-judgement` to recompute literal `Yes`
and `No` logits with one backend,
then holds those logits fixed while comparing the historical candidate fallback
with full binary normalization. Row-level logits are optional large diagnostic
artifacts and are not included; `judgement_scoring_rule_audit.csv` contains the
reported per-cell audit. The released command accepts the row-level JSON/JSONL
schema when those artifacts are available.

The same boundary applies to correctness parsing, Yes/No self-judgement
elicitation, and the X/Y counterbalanced OOD elicitation. The counterbalanced
release contains the mapping-corrected self-judgement probability and exact
retained candidates, but not the four raw XY/YX option log-probabilities for
each pre-filter candidate. Activation extraction, answer-likelihood scoring,
thresholding from the released balanced probabilities, and all reported
activation analyses are rerunnable. This keeps
the artifact faithful to the executed study instead of presenting a newly
written generation pipeline as historical code.

## Data-construction robustness

The repeated Qwen2.5-7B strict-pair sample is released under
`data/processed/robustness/qwen25_7b_r2/`. Its `all_pairs.jsonl` files contain
the second draw after full Yes/No judgement scoring; `strict_pairs.jsonl`
applies the primary paired threshold at `tau=0.7` and can be passed directly to
`metacog-extract`. Given corresponding activation artifacts, downstream Exp2B
estimates are reconstructed with the ordinary analysis commands. Given the
optional frozen pass-at-eight pools,
`metacog-robustness resample-pairs` repeats the seed-2027 draw uniformly over
valid generation slots, leaving the original response eligible and preserving
duplicate-slot multiplicity.

`data/processed/robustness/movies_multireference_question_ids.txt` contains the
159 stable hashes of exact Movies prompts associated with multiple reference
actors. `metacog-robustness filter-multireference` removes those rows from any
released Movies record file before activation subsetting or re-extraction.
The publication tables contain the reported sample flow and Exp2B comparisons.

## Activation schema

Each model/domain directory contains:

```text
metadata.json
samples.jsonl
layers_<first>_<last>.npz
checksums.json
```

The NPZ file stores `activations` with shape
`(n_samples, n_layers, hidden_size)`, integer `layers`, and aligned `sample_id`,
`objective_correctness`, `self_judgement`, and `p_self_judgement` arrays.
Activations are residual-stream states at the output of each complete
transformer block, evaluated at the final non-padding answer token. Forward
passes use bfloat16 by default and arrays are stored as float16.

Historical activation metadata did not preserve immutable model-repository
snapshot revisions. Accordingly, `configs/models.yaml` leaves each `revision`
field unset rather than assigning an unverified value. The released numerical
artifacts make analyses of the existing activations auditable, but a fresh run
that resolves a moving model identifier may not be byte-identical. New
extraction runs should record the resolved checkpoint revision alongside the
activation checksums.

## Statistical settings

- Primary confidence threshold: symmetric `tau=0.7`.
- Exp2A: five-fold `GroupKFold` by exact question identifier.
- Primary layer window: normalized depth `[0.40, 0.80]`.
- Primary intervals: 1,000 question-cluster bootstrap repetitions.
- Threshold sensitivity: 100 repetitions.
- Exp2B primary inference: independent source and target question resampling,
  with directions refitted in every replicate.
- Analysis-specific seeds are recorded in `configs/analysis.yaml`; a single
  global seed is not substituted for the seeds used by the archived runs.

`metacog-analyze exp2a-window` computes the grouped out-of-fold fixed-window
endpoint. `metacog-analyze exp2b-window` and `ood-window` hold the fitted source
directions fixed and bootstrap target questions. The primary ID interval uses
`joint-exp2b`, which independently resamples source and target questions and
refits every source direction in each replicate. The supplementary
`source-question-fe` analysis first differences each source question's paired
correct and incorrect activations, estimates the OC, SJ, and interaction
coefficient vectors, and repeats the same joint source-target bootstrap.

The activation metadata field `token_count`, used by B/C matching, is the complete
rendered question-plus-answer sequence length. The free-response likelihood
sidecar separately records `answer_token_count`, the number of scored assistant
answer tokens. For forced-choice OOD controls, `token_count` is again the complete
rendered question-plus-answer sequence length. The sidecar also records
option-normalized selected-answer log probability and non-reference answer-letter
indicators.

`examples/score_answer_likelihood.sh` generates this sidecar. The exact
letter-adjusted OOD analysis is shown in
`examples/run_ood_nuisance_analysis.sh`: MMLU includes indicators for B, C, and
D with A as reference, while binary TruthfulQA includes B with A as reference.
