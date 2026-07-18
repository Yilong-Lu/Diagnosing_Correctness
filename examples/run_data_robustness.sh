#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/robustness}"
mkdir -p "${OUTPUT_ROOT}"

# The released hash list is sufficient to reproduce the Movies row exclusion.
metacog-robustness filter-multireference \
  --question-ids data/processed/robustness/movies_multireference_question_ids.txt \
  --input data/processed/id/qwen25_7b/movies/strict_pairs.jsonl \
  --output "${OUTPUT_ROOT}/qwen25_7b_movies_filtered.jsonl"

# Optional GPU scoring recreates the row-level fixed-logit audit input.
if [[ -n "${MODEL_PATH:-}" && -n "${JUDGEMENT_INPUT:-}" ]]; then
  metacog-score-judgement \
    --model "${MODEL_PATH}" \
    --domain "${DOMAIN:-math}" \
    --input "${JUDGEMENT_INPUT}" \
    --output "${OUTPUT_ROOT}/judgement_score_rows.jsonl" \
    --local-files-only
  JUDGEMENT_SCORE_ROWS="${OUTPUT_ROOT}/judgement_score_rows.jsonl"
fi

# Existing row-level score artifacts can also be summarized directly.
if [[ -n "${JUDGEMENT_SCORE_ROWS:-}" ]]; then
  metacog-robustness score-audit \
    --input "${JUDGEMENT_SCORE_ROWS}" \
    --output "${OUTPUT_ROOT}/judgement_score_audit.json"
fi

# The released R2 strict-pair records can be passed directly to metacog-extract.
# Given optional pass-at-eight pools, the draw itself can also be repeated:
# metacog-robustness resample-pairs \
#   --pool /path/to/frozen_pool.json \
#   --original data/processed/id/qwen25_7b/math/all_pairs.jsonl \
#   --domain math --seed 2027 \
#   --output "${OUTPUT_ROOT}/math_r2_pairs.jsonl" \
#   --summary "${OUTPUT_ROOT}/math_r2_summary.json"
