# Third-Party Materials

The MIT license at the repository root applies only to the code written for this
artifact. It does not relicense benchmark text, model checkpoints, tokenizers,
or upstream repositories.

## Current release package

The tracked package contains aggregate numerical source-data tables and the
processed response records required to reconstruct the reported activation
inputs. It contains no model weights or hidden states. The processed files add
model responses, correctness labels, self-judgement probabilities, stable
identifiers, and filtering metadata to upstream question text.

## External requirements

- Model checkpoints and tokenizers must be obtained from their official model
  distributors under the applicable model licenses.
- Movies originates from the repository accompanying *LLMs Know More Than They
  Show*, distributed under MIT terms.
- MMLU originates from the official `hendrycks/test` distribution under MIT
  terms.
- TruthfulQA originates from the official `sylinrl/TruthfulQA` distribution
  under Apache-2.0 terms.
- The Math pool contains the source tags `amc`, `math`, `amc2023`, `math500`,
  `gsm8k`, `SVAMP`, `ASDiv`, and `MultiArith`. The `amc` and `math` imports came
  through the MIT-licensed DeepScaleR repository; GSM8K is MIT licensed. The
  remaining word-problem imports are retained with source tags because their
  historical acquisition snapshot did not preserve an immutable upstream
  revision or a single common license. They must not be described as covered
  by the project code's MIT license.

The release manifest provides checksums for the exact transformed files.
Upstream benchmark and model terms continue to govern redistribution and use of
their respective materials.
