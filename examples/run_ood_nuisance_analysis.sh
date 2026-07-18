#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${MODEL_KEY:-qwen25_7b}"
SOURCE_DOMAIN="${SOURCE_DOMAIN:-math}"
TARGET_DOMAIN="${TARGET_DOMAIN:-mmlu}"
ACTIVATION_ROOT="${ACTIVATION_ROOT:-activations}"
SIDECAR_ROOT="${SIDECAR_ROOT:-sidecars}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis}"

SOURCE="${ACTIVATION_ROOT}/${MODEL_KEY}/${SOURCE_DOMAIN}"
TARGET="${ACTIVATION_ROOT}/${MODEL_KEY}/${TARGET_DOMAIN}"
SIDECAR="${SIDECAR_ROOT}/${MODEL_KEY}/${TARGET_DOMAIN}/answer_likelihood.csv"
OUTPUT="${OUTPUT_ROOT}/${MODEL_KEY}/${SOURCE_DOMAIN}_to_${TARGET_DOMAIN}_nuisance.json"

NUISANCE_COLUMNS=(mean_answer_logprob token_count)
if [[ "${TARGET_DOMAIN}" == "mmlu" ]]; then
  NUISANCE_COLUMNS+=(answer_letter_B answer_letter_C answer_letter_D)
elif [[ "${TARGET_DOMAIN}" == "truthfulqa_binary" ]]; then
  NUISANCE_COLUMNS+=(answer_letter_B)
else
  echo "TARGET_DOMAIN must be mmlu or truthfulqa_binary" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT}")"
metacog-analyze nuisance \
  --source "${SOURCE}" \
  --target "${TARGET}" \
  --nuisance-csv "${SIDECAR}" \
  --nuisance-columns "${NUISANCE_COLUMNS[@]}" \
  --output "${OUTPUT}" \
  --bootstrap 1000
