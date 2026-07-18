# Numerical Validation

The cleaned implementation was compared against the frozen manuscript source
data using the local activation artifacts. Validation uses no language-model
calls and does not modify the activation files.

## Completed checks

- Reconstructing the symmetric `tau=0.7` strict sample from all judged pairs
  reproduces the reported row, pair, and unique-question counts for all four
  models and both source domains.
- Qwen2.5-7B Math-to-Movies Exp1 point estimates match the archived all-layer
  AUC values exactly at all 28 layers.
- Qwen2.5-7B Math-to-Movies Exp2B component and relative AUC point estimates
  match the archived values exactly at all 28 layers.
- Qwen2.5-7B Math grouped Exp2A point estimates agree with the archived
  question-grouped implementation to floating-point precision.
- The Qwen2.5-7B Math-to-Movies fixed-window Exp2B point estimates and a
  ten-repetition joint source-target bootstrap regression check match the
  archived implementation to machine precision.
- The fixed-logit scoring-rule table contains no SJ label or paired-threshold
  membership changes, with maximum row-wise difference below `1.31e-6`.
- Released Qwen2.5-7B R2 record counts match the reported 2,289 Math and 907
  Movies strict pairs; both fixed-window intervals remain above zero.
- The Movies multi-reference release contains 159 unique prompt hashes, and all
  eight filtered fixed-window intervals remain above zero.

The reduced-bootstrap check is a regression test of implementation identity,
not a replacement for the 1,000-repetition manuscript run. Publication tables
in `artifacts/publication` retain the full-run results.
