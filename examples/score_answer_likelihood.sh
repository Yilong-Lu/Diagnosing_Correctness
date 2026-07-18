#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${MODEL_KEY:-qwen25_7b}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DOMAIN="${DOMAIN:-math}"
if [[ "${DOMAIN}" == "math" || "${DOMAIN}" == "movies" ]]; then
  DEFAULT_INPUT="data/processed/id/${MODEL_KEY}/${DOMAIN}/strict_pairs.jsonl"
else
  DEFAULT_INPUT="data/processed/ood/${MODEL_KEY}/${DOMAIN}/strict_conflicts.jsonl"
fi
INPUT="${INPUT:-${DEFAULT_INPUT}}"
OUTPUT="${OUTPUT:-sidecars/${MODEL_KEY}/${DOMAIN}/answer_likelihood.csv}"
BATCH_SIZE="${BATCH_SIZE:-2}"

MODE="free-response"
CHOICES="A,B,C,D"
if [[ "${DOMAIN}" == "mmlu" || "${DOMAIN}" == "truthfulqa_binary" ]]; then
  MODE="forced-choice"
fi
if [[ "${DOMAIN}" == "truthfulqa_binary" ]]; then
  CHOICES="A,B"
fi

metacog-score-answer \
  --mode "${MODE}" \
  --model "${MODEL_PATH}" \
  --domain "${DOMAIN}" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --choices "${CHOICES}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --device-dtype bfloat16
