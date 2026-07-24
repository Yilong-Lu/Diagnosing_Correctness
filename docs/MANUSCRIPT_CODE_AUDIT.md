# Manuscript-to-Code Completeness Audit

This ledger maps each reported analysis to its activation-level entry point and
released numerical source, while distinguishing estimation from
presentation-only rendering.

| Reported evidence | Activation-level entry point | Released numerical source | Status |
| --- | --- | --- | --- |
| Exp1 mixed-direction diagnostic | `metacog-analyze exp1` | `id_all_layer_question_cluster_intervals.csv` | Complete |
| Exp2A grouped all-layer curves | `metacog-analyze exp2a` | `exp2a_grouped_all_layers.csv` | Complete |
| Exp2A grouped window endpoint | `metacog-analyze exp2a-window` | `exp2a_grouped_window.csv` | Complete |
| Exp2B all-layer transfer | `metacog-analyze exp2b` | `id_all_layer_question_cluster_intervals.csv` | Complete |
| Exp2B primary joint interval | `metacog-analyze joint-exp2b` | `exp2b_joint_source_target_window.csv` | Complete |
| Descriptive peak summary | `metacog-summarize exp2b-peaks` | `exp2b_peak_summary.csv` | Complete |
| OC-only mass-mean control | `metacog-analyze oc-only` | `oc_only_window.csv` | Complete |
| Target-question fixed effects and within-component specificity | `metacog-analyze question-fe` | `question_fe_contrast.csv`, `question_fe_coefficients.csv` | Complete |
| B/C token-count matching | `metacog-analyze token-match` | `bc_matching_diagnostics.csv` | Complete; use `--matching-bins 10` for fixed deciles |
| Label-shuffle and random-direction nulls | `metacog-analyze null-controls` | `window_null_controls.csv` | Complete |
| ID answer-likelihood control | `metacog-score-answer`; `metacog-analyze nuisance` | `main_answer_logp_controls.csv` | Complete |
| Symmetric-threshold sensitivity | `metacog-prepare --kind rethreshold-id`; layer analysis commands | `threshold_counts.csv`, `threshold_exp2b_all_layers.csv` | Complete from released all-pairs records; default filtering is row-level |
| OOD all-layer and fixed-window transfer | `metacog-analyze ood`; `ood-window` | `ood_all_layer_transfer.csv` | Complete |
| OOD answer-likelihood and answer-letter control | `metacog-score-answer --mode forced-choice`; `metacog-analyze nuisance`; `examples/run_ood_nuisance_analysis.sh` | `ood_answer_logp_controls.csv` | Complete |
| OOD X/Y counterbalanced analysis | ordinary extraction and OOD analysis on `data/processed/ood_counterbalanced` | `ood_counterbalanced_controls.csv`, `ood_counterbalanced_diagnostics.csv` | Complete from released mapping-corrected probabilities and retained candidates |
| OOD heterogeneity counts | `metacog-summarize ood-heterogeneity` | `ood_heterogeneity.csv` | Complete |
| Fixed-logit judgement scoring-rule audit | `metacog-score-judgement`; `metacog-robustness score-audit` | `judgement_scoring_rule_audit.csv` | Complete scoring and estimator code with the aggregate audit included |
| Qwen2.5-7B strict-pair resampling | `metacog-robustness resample-pairs`; ordinary extraction and Exp2B commands | `data/processed/robustness/qwen25_7b_r2`, `qwen25_7b_r2_sample_flow.csv`, `qwen25_7b_r2_exp2b.csv` | Complete from released R2 records; the draw can also be repeated from response pools |
| Movies multi-reference prompt sensitivity | `metacog-robustness filter-multireference`; ordinary Exp2B commands | `movies_multireference_question_ids.txt`, `movies_multireference_sample_flow.csv`, `movies_multireference_exp2b.csv` | Complete from released hashes, records, and schema-compatible activations |
| Figures 2--4 and data-driven supplementary figures | `metacog-reproduce` | all-layer CSV files and released pre-filter probabilities | Complete |

## Reproduction starting point

The package begins from the exact processed response and self-judgement records
used in the study. These records fix the outputs of question selection, response
generation, correctness parsing, judgement elicitation, and response-pair
sampling. Every stage from these records through activation extraction and
reported statistics is represented by released code. Hidden-state arrays are
excluded from Git because of their size and can be regenerated from the listed
public checkpoints or supplied as schema-compatible artifacts. The released
Qwen2.5-7B R2 records contain the realized second response-pair draw, and the
accompanying command can repeat that draw when response pools are supplied.
