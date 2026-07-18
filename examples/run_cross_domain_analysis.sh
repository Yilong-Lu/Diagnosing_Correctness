#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${MODEL_KEY:-qwen25_7b}"
SOURCE_DOMAIN="${SOURCE_DOMAIN:-math}"
TARGET_DOMAIN="${TARGET_DOMAIN:-movies}"
ACTIVATION_ROOT="${ACTIVATION_ROOT:-activations}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis}"

SOURCE="${ACTIVATION_ROOT}/${MODEL_KEY}/${SOURCE_DOMAIN}"
TARGET="${ACTIVATION_ROOT}/${MODEL_KEY}/${TARGET_DOMAIN}"
mkdir -p "${OUTPUT_ROOT}/${MODEL_KEY}"

metacog-analyze exp2b \
  --source "${SOURCE}" \
  --target "${TARGET}" \
  --output "${OUTPUT_ROOT}/${MODEL_KEY}/${SOURCE_DOMAIN}_to_${TARGET_DOMAIN}_layers.csv" \
  --bootstrap 1000

metacog-analyze joint-exp2b \
  --source "${SOURCE}" \
  --target "${TARGET}" \
  --output "${OUTPUT_ROOT}/${MODEL_KEY}/${SOURCE_DOMAIN}_to_${TARGET_DOMAIN}_joint.json" \
  --bootstrap 1000

metacog-analyze source-question-fe \
  --source "${SOURCE}" \
  --target "${TARGET}" \
  --output "${OUTPUT_ROOT}/${MODEL_KEY}/${SOURCE_DOMAIN}_to_${TARGET_DOMAIN}_source_question_fe.json" \
  --bootstrap 1000

metacog-analyze exp2b-window \
  --source "${SOURCE}" \
  --target "${TARGET}" \
  --output "${OUTPUT_ROOT}/${MODEL_KEY}/${SOURCE_DOMAIN}_to_${TARGET_DOMAIN}_fixed_source_window.json" \
  --bootstrap 1000
