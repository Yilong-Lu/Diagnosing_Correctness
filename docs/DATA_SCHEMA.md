# Processed Record Schema

`metacog-prepare` converts frozen judged-response files into a common JSON Lines
schema. The command never calls a language model.

## Main-domain pairs

```bash
metacog-prepare \
  --kind id-pairs \
  --input judged_pairs_all.json \
  --output data/processed/id/qwen25_7b/math \
  --model qwen25_7b \
  --domain math \
  --threshold 0.7
```

The input must contain consecutive two-row pairs with the same question and one
correct and one incorrect response. The command writes `all_pairs.jsonl` and a
pair-filtered `strict_pairs.jsonl`.

Released all-pairs records can be re-thresholded without the historical JSON
input. By default this reproduces the threshold-sensitivity analysis: individual
responses in the uncertain band are removed without imposing an additional
pair-completeness rule.

```bash
metacog-prepare \
  --kind rethreshold-id \
  --input data/processed/id/qwen25_7b/math/all_pairs.jsonl \
  --output outputs/qwen25_7b_math_tau06.jsonl \
  --model qwen25_7b \
  --domain math \
  --threshold 0.6
```

Add `--pairwise` to require both members of every response pair to pass. That
pairwise rule is used to build the primary `strict_pairs.jsonl` files at
`tau=0.7`.

## OOD candidates

```bash
metacog-prepare \
  --kind ood-candidates \
  --input judged_candidates.jsonl \
  --output data/processed/ood/qwen25_7b/mmlu \
  --model qwen25_7b \
  --domain mmlu \
  --threshold 0.7
```

OOD inputs may provide ordinary `p_judgement` or the balanced probability from
the X/Y elicitation control. The command writes all candidates, confident
candidates, and the confident B/C conflict subset used for activation transfer.

## Common fields

| Field | Meaning |
| --- | --- |
| `sample_id` | Row index in the processed file |
| `model` | Public model key |
| `domain` | Public domain key |
| `pair_id` | Response-pair or OOD-question identifier |
| `question_id` | Stable exact-question cluster identifier |
| `objective_correctness` | Binary dataset/parser correctness label |
| `p_self_judgement` | Continuous probability assigned to “correct” |
| `self_judgement` | Thresholded high-confidence label |
| `threshold_keep` | Whether the row satisfies the confidence rule |
| `threshold_mode` | `symmetric_confident` |

Activation-artifact sample indices additionally store `token_count`, defined as
the token length of the complete rendered question-plus-answer sequence. The
free-response answer-likelihood sidecar uses the distinct field
`answer_token_count` for the scored assistant-answer span.

## Robustness records

The Qwen2.5-7B R2 directories use the same main-domain schema, so their
`strict_pairs.jsonl` files are ordinary activation-extraction inputs. The
multi-reference sensitivity releases only stable exact-question hashes, one per
line. The filter command accepts either this hash list or a source JSON file
from which the set can be derived.
