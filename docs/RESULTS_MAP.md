# Results-to-Code Map

| Paper component | Reproduction module | Required input |
| --- | --- | --- |
| Exp1 mixed-direction conflict diagnostic | `metacog.experiments.exp1` | Math/Movies activations |
| Exp2A all-layer and fixed-window validation | `metacog-analyze exp2a`; `exp2a-window` | Math/Movies activations |
| Exp2B all-layer and fixed-source window transfer | `metacog-analyze exp2b`; `exp2b-window` | Math/Movies activations |
| Joint source-target bootstrap | `metacog-analyze joint-exp2b` | Math/Movies activations |
| OC-only mass-mean control | `metacog-analyze oc-only` | Math/Movies activations |
| Source-question-adjusted directions | `metacog-analyze source-question-fe` | Math/Movies activations |
| Target-question fixed effects and within-component specificity | `metacog-analyze question-fe` | Math/Movies activations |
| Answer-likelihood and OOD answer-letter residualization | `metacog-score-answer`; `metacog-analyze nuisance`; `examples/run_ood_nuisance_analysis.sh` | Activations and generated sidecar |
| B/C token-count matching | `metacog-analyze token-match` | Math/Movies activations |
| Label-shuffle and random-direction nulls | `metacog-analyze null-controls` | Math/Movies activations |
| MMLU and TruthfulQA transfer | `metacog-analyze ood`; `ood-window` | Source and OOD activations |
| Descriptive peak and OOD heterogeneity summaries | `metacog-summarize` | All-layer result CSVs |
| Fixed-logit Yes/No scoring-rule audit | `metacog-score-judgement`; `metacog-robustness score-audit` | Per-row recomputed score records; released aggregate in `judgement_scoring_rule_audit.csv` |
| Qwen2.5-7B second strict-pair draw | `metacog-robustness resample-pairs`; ordinary extraction and Exp2B commands | Optional frozen pass-at-eight pools; released R2 records and `qwen25_7b_r2_*.csv` |
| Movies multi-reference sensitivity | `metacog-robustness filter-multireference`; ordinary Exp2B commands | Released prompt hashes, processed Movies records, and `movies_multireference_*.csv` |
| Main figures and tables | `metacog.cli.reproduce` | Publication CSV artifacts |

The conceptual Figure 1 is not data-generated. Figures 2--4 are regenerated
from the included numerical source data. Publication tables are copied as
auditable CSV source data; their estimators are exposed by the analysis commands
above, while manuscript-specific LaTeX rendering is intentionally separate from
the statistical implementation. See `docs/MANUSCRIPT_CODE_AUDIT.md` for the
claim-level completeness audit.
