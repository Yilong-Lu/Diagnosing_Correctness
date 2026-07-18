"""Answer-likelihood helpers shared by main-domain and forced-choice controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def align_response_tokens(prefix_ids: Sequence[int], full_ids: Sequence[int]) -> tuple[list[int], int]:
    """Mask prompt tokens while tolerating at most two tokenizer boundary merges."""

    prefix = list(prefix_ids)
    full = list(full_ids)
    if len(full) >= len(prefix) and full[: len(prefix)] == prefix:
        start = len(prefix)
    else:
        start = 0
        for left, right in zip(prefix, full):
            if left != right:
                break
            start += 1
        if start < max(1, len(prefix) - 2):
            raise ValueError(
                "could not align the answer prompt with the complete answer sequence: "
                f"common={start}, prefix={len(prefix)}, full={len(full)}"
            )
    if start >= len(full):
        raise ValueError("answer sequence contains no response tokens")
    return [-100] * start + full[start:], start


def token_ids_for_choice(tokenizer, choice: str) -> list[int]:
    """Return token IDs for common single-token renderings of an answer choice."""

    single: list[int] = []
    fallback: list[int] = []
    for variant in (choice, " " + choice, "\n" + choice):
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) == 1:
            single.append(int(encoded[0]))
        elif encoded:
            fallback.append(int(encoded[-1]))
    return sorted(set(single or fallback))


def forced_choice_row(
    *,
    sample_id: int,
    selected: str,
    choice_logprobs: Mapping[str, float],
    choices: Sequence[str],
    token_count: int,
) -> dict[str, float | int]:
    """Build the nuisance sidecar row for one forced-choice candidate."""

    ordered = [str(choice) for choice in choices]
    if selected not in choice_logprobs:
        raise ValueError(f"selected option {selected!r} was not scored")
    values = np.asarray([float(choice_logprobs[choice]) for choice in ordered])
    normalizer = float(np.logaddexp.reduce(values))
    row: dict[str, float | int] = {
        "sample_id": int(sample_id),
        "mean_answer_logprob": float(choice_logprobs[selected]) - normalizer,
        "raw_answer_logprob": float(choice_logprobs[selected]),
        "token_count": int(token_count),
    }
    reference = ordered[0]
    for choice in ordered[1:]:
        row[f"answer_letter_{choice}"] = float(selected == choice)
    row["answer_letter_reference"] = reference
    return row
