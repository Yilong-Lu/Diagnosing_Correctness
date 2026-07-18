"""Literal Yes/No next-token scoring for self-judgement robustness checks."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .prompts import answer_prompt


JUDGEMENT_PROMPT = "Do you believe the answer above is correct? Answer only with Yes or No."


def build_judgement_messages(record: Mapping[str, Any], domain: str) -> list[dict[str, str]]:
    """Reconstruct the three-message self-judgement conversation."""

    return [
        {"role": "user", "content": answer_prompt(dict(record), domain)},
        {
            "role": "assistant",
            "content": str(record.get("response", record.get("answer", ""))),
        },
        {"role": "user", "content": JUDGEMENT_PROMPT},
    ]


def render_judgement_prompt(tokenizer, record: Mapping[str, Any], domain: str) -> str:
    return tokenizer.apply_chat_template(
        build_judgement_messages(record, domain),
        tokenize=False,
        add_generation_prompt=True,
    )


def resolve_literal_token_ids(tokenizer) -> dict[str, int]:
    """Require literal Yes and No to each be one tokenizer token."""

    token_ids = {}
    for candidate in ("Yes", "No"):
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"literal candidate {candidate!r} must encode as one token; got {encoded}"
            )
        token_ids[candidate] = int(encoded[0])
    if token_ids["Yes"] == token_ids["No"]:
        raise ValueError("Yes and No resolved to the same token ID")
    return token_ids


def classify_topk_coverage(
    token_ids: Sequence[int], *, yes_token_id: int, no_token_id: int
) -> str:
    present = {int(token_id) for token_id in token_ids}
    has_yes = yes_token_id in present
    has_no = no_token_id in present
    if has_yes and has_no:
        return "both"
    if has_yes:
        return "yes_only"
    if has_no:
        return "no_only"
    return "neither"


def full_binary_probability(logit_yes: float, logit_no: float) -> float:
    """Compute sigmoid(logit_yes - logit_no) without overflow."""

    difference = float(logit_yes) - float(logit_no)
    if difference >= 0:
        return 1.0 / (1.0 + math.exp(-difference))
    exp_difference = math.exp(difference)
    return exp_difference / (1.0 + exp_difference)


def historical_top4_probability(logp_yes: float, logp_no: float, coverage: str) -> float:
    """Apply the historical top-four fallback to fixed next-token log probabilities."""

    if coverage == "both":
        return full_binary_probability(logp_yes, logp_no)
    if coverage == "yes_only":
        return math.exp(float(logp_yes))
    if coverage == "no_only":
        return 1.0 - math.exp(float(logp_no))
    if coverage == "neither":
        return 0.5
    raise ValueError(f"unsupported coverage category: {coverage}")


def _next_token_logits(model, tensors: Mapping[str, Any]):
    """Request only final-position logits when the model API supports it."""

    parameters = inspect.signature(model.forward).parameters
    kwargs = dict(tensors)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    elif "num_logits_to_keep" in parameters:
        kwargs["num_logits_to_keep"] = 1
    return model(**kwargs).logits[:, -1, :]


def score_judgement_logits(
    model,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> list[dict[str, Any]]:
    """Score literal Yes/No logits at the first assistant-generation position."""

    import torch
    from tqdm import tqdm

    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must have a pad token before scoring")
    token_ids = resolve_literal_token_ids(tokenizer)
    prompts = [render_judgement_prompt(tokenizer, row, domain) for row in records]
    encoded = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    too_long = [index for index, ids in enumerate(encoded) if len(ids) > max_length]
    if too_long:
        raise ValueError(
            f"{len(too_long)} judgement prompts exceed --max-length; first index {too_long[0]}"
        )

    output: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(encoded), batch_size), desc="judgement-logit batches"):
        batch = [{"input_ids": ids} for ids in encoded[start : start + batch_size]]
        tensors = tokenizer.pad(batch, padding=True, return_tensors="pt")
        tensors = {name: value.to(device) for name, value in tensors.items()}
        with torch.inference_mode():
            logits = _next_token_logits(model, tensors).float()
            logprobs = torch.log_softmax(logits, dim=-1)
            _, top_ids = torch.topk(logits, k=4, dim=-1)

        for local_index in range(logits.shape[0]):
            row_logits = logits[local_index]
            row_logprobs = logprobs[local_index]
            yes_logit = float(row_logits[token_ids["Yes"]].item())
            no_logit = float(row_logits[token_ids["No"]].item())
            yes_logp = float(row_logprobs[token_ids["Yes"]].item())
            no_logp = float(row_logprobs[token_ids["No"]].item())
            top = [int(value) for value in top_ids[local_index].tolist()]
            coverage = classify_topk_coverage(
                top,
                yes_token_id=token_ids["Yes"],
                no_token_id=token_ids["No"],
            )
            record = records[start + local_index]
            correctness = record.get("correct", record.get("objective_correctness"))
            if correctness is None:
                raise ValueError("record lacks a correctness field")
            output.append(
                {
                    "row_index": start + local_index,
                    "question": str(record["question"]),
                    "correct": bool(correctness),
                    "logit_yes": yes_logit,
                    "logit_no": no_logit,
                    "logp_yes": yes_logp,
                    "logp_no": no_logp,
                    "coverage": coverage,
                    "top4_token_ids": top,
                    "p_judgement_reconstructed": historical_top4_probability(
                        yes_logp, no_logp, coverage
                    ),
                    "p_judgement_full": full_binary_probability(yes_logit, no_logit),
                }
            )
    return output
