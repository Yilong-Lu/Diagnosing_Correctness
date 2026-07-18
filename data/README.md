# Data Package

This directory contains the frozen processed records used to extract the
reported activations. It does not contain model weights or hidden-state arrays.

```text
processed/id/<model>/<domain>/all_pairs.jsonl
processed/id/<model>/<domain>/strict_pairs.jsonl
processed/ood/<model>/<target>/strict_conflicts.jsonl
processed/ood_counterbalanced/<model>/<target>/strict_conflicts.jsonl
processed/robustness/qwen25_7b_r2/<domain>/{all_pairs,strict_pairs}.jsonl
processed/robustness/movies_multireference_question_ids.txt
```

`all_pairs.jsonl` supports reconstruction of the symmetric-threshold sensitivity
samples. `strict_pairs.jsonl` is the paired `tau=0.7` input to the primary Math
and Movies activation pass. OOD files contain the exact high-confidence B/C
candidates whose answer-letter activations were evaluated. Sample identifiers
in activation inputs are contiguous and therefore align with generated
activation rows and answer-likelihood sidecars.

The robustness directory contains the judged second Qwen2.5-7B strict-pair draw
and the 159 stable prompt hashes used by the Movies multi-reference exclusion.
These files support the downstream analyses in Supplementary Sec. S8.7 without
requiring the optional frozen pass-at-eight response pools.

`manifest.csv` records every released file's row count, byte size, SHA-256
checksum, source family, and applicable terms. `math_source_composition.csv`
records the eight source tags in the frozen Math pool. These processed records
include model-generated responses and judgements; upstream benchmark terms
continue to apply to embedded question text. See `docs/THIRD_PARTY.md`.
