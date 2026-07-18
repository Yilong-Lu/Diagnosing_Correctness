# Publication Artifacts

This directory contains lightweight numerical source data used to rebuild the
paper's data figures and tables. These files contain aggregate estimates and
confidence intervals, not benchmark questions, model responses, hidden states,
usernames, machine paths, or scheduler metadata.

`manifest.json` records each file's role, byte size, and SHA-256 checksum. Public
identifiers are normalized across files:

- `math` for the mathematical-reasoning domain;
- `truthfulqa_binary` for binary TruthfulQA;
- `llama31_8b` for Llama-3.1-8B-Instruct.

The publication artifacts support CPU-only reconstruction of Figures 2--4 and
provide the numerical source data for main and supplementary tables, including
threshold sensitivity, matching, null, fixed-effect, likelihood, and OOD
controls. The counterbalanced OOD all-layer table supports reconstruction of
the corresponding supplementary curve figure. Additional tables record the
fixed-logit judgement-scoring audit, Qwen2.5-7B strict-pair resampling, and
Movies multi-reference sensitivity reported in Supplementary Sec. S8.7. These
artifacts do not replace activation-level recomputation.
