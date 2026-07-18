#!/usr/bin/env bash
set -euo pipefail

# Override these variables for another checkpoint, model, or domain. MODEL_PATH
# may be a public model ID or a local checkpoint directory.
MODEL_KEY="${MODEL_KEY:-qwen25_7b}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DOMAIN="${DOMAIN:-math}"
if [[ "${DOMAIN}" == "math" || "${DOMAIN}" == "movies" ]]; then
  DEFAULT_INPUT="data/processed/id/${MODEL_KEY}/${DOMAIN}/strict_pairs.jsonl"
else
  DEFAULT_INPUT="data/processed/ood/${MODEL_KEY}/${DOMAIN}/strict_conflicts.jsonl"
fi
INPUT="${INPUT:-${DEFAULT_INPUT}}"
OUTPUT="${OUTPUT:-activations}"
BATCH_SIZE="${BATCH_SIZE:-2}"

metacog-extract \
  --model-key "${MODEL_KEY}" \
  --model "${MODEL_PATH}" \
  --domain "${DOMAIN}" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --layers all \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --device-dtype bfloat16 \
  --storage-dtype float16
