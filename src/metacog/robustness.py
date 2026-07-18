"""Reusable data-construction robustness checks reported in the supplement."""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


NUMBER_PATTERN = re.compile(r"^\d+(\.\d+)?$")


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a deterministic local seed without depending on Python's hash seed."""

    digest = hashlib.blake2s(
        "::".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little")


def is_symmetric_confident(probability: float, threshold: float = 0.7) -> bool:
    """Return whether a probability lies outside the open uncertainty band."""

    value = float(probability)
    return value <= 1.0 - threshold or value >= threshold


def _correctness(row: Mapping[str, Any]) -> bool:
    if "correct" in row:
        return bool(row["correct"])
    if "objective_correctness" in row:
        return bool(row["objective_correctness"])
    raise ValueError("row lacks a correctness field")


def attach_adjacent_pair_ids(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate adjacent strict pairs and attach an integer audit pair ID."""

    if len(rows) % 2:
        raise ValueError("strict-pair rows must have even length")
    output: list[dict[str, Any]] = []
    for start in range(0, len(rows), 2):
        left, right = rows[start], rows[start + 1]
        if str(left["question"]) != str(right["question"]):
            raise ValueError(f"rows {start} and {start + 1} do not contain the same question")
        if {_correctness(left), _correctness(right)} != {False, True}:
            raise ValueError(
                f"rows {start} and {start + 1} must contain one correct and one incorrect response"
            )
        output.append({**left, "audit_pair_id": start // 2})
        output.append({**right, "audit_pair_id": start // 2})
    return output


def complete_pair_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    threshold: float = 0.7,
) -> set[int]:
    """Return pairs for which both responses satisfy symmetric confidence."""

    flags: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        flags[int(row["audit_pair_id"])].append(
            is_symmetric_confident(float(row[score_key]), threshold)
        )
    return {pair_id for pair_id, values in flags.items() if len(values) == 2 and all(values)}


def summarize_scoring_rule(
    rows: Sequence[Mapping[str, Any]],
    *,
    historical_key: str,
    full_key: str,
    threshold: float = 0.7,
) -> dict[str, float | int]:
    """Compare two judgement scores while holding their underlying logits fixed."""

    paired = attach_adjacent_pair_ids(rows)
    historical = np.asarray([float(row[historical_key]) for row in paired])
    full = np.asarray([float(row[full_key]) for row in paired])
    historical_pairs = complete_pair_ids(
        paired, score_key=historical_key, threshold=threshold
    )
    full_pairs = complete_pair_ids(paired, score_key=full_key, threshold=threshold)
    difference = np.abs(historical - full)
    return {
        "n_rows": len(paired),
        "n_pairs": len(paired) // 2,
        "mae": float(np.mean(difference)),
        "max_abs_error": float(np.max(difference)),
        "label_flip_n": int(np.sum((historical > 0.5) != (full > 0.5))),
        "historical_complete_pair_n": len(historical_pairs),
        "full_complete_pair_n": len(full_pairs),
        "complete_pair_membership_change_n": len(
            historical_pairs.symmetric_difference(full_pairs)
        ),
    }


def _validate_parallel_fields(item: Mapping[str, Any], fields: Sequence[str]) -> int:
    lengths = {field: len(item[field]) for field in fields}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"response-pool fields have inconsistent lengths: {lengths}")
    return next(iter(lengths.values()))


def candidate_pool(item: Mapping[str, Any], domain: str) -> list[dict[str, Any]]:
    """Apply the study's response-validity rules to one frozen pass-at-eight pool."""

    fields = ["outputs", "answers", "correctness"]
    if domain == "movies" and "validity" in item:
        fields.append("validity")
    n = _validate_parallel_fields(item, fields)
    validity = item.get("validity", [True] * n)
    candidates: list[dict[str, Any]] = []
    for index, (response, answer, correct, valid) in enumerate(
        zip(item["outputs"], item["answers"], item["correctness"], validity)
    ):
        response = str(response)
        answer = str(answer)
        if domain == "math":
            if not response or response[-1] not in "0123456789. \n":
                continue
            if response.count("Answer:") != 1 or not NUMBER_PATTERN.fullmatch(answer):
                continue
            if not bool(correct) and "none" in answer.lower():
                continue
        elif domain == "movies":
            if not bool(valid):
                continue
        else:
            raise ValueError(f"unsupported domain: {domain}")
        candidates.append(
            {
                "response": response,
                "answer": answer,
                "correct": bool(correct),
                "pass_index": index,
            }
        )
    return candidates


def _group_original_pairs(
    records: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    paired = attach_adjacent_pair_ids(records)
    pairs: list[list[dict[str, Any]]] = []
    for start in range(0, len(paired), 2):
        pair = [dict(paired[start]), dict(paired[start + 1])]
        pair.sort(key=lambda row: not _correctness(row))
        pairs.append(pair)
    return pairs


def _pair_matches_pool(
    pair: Sequence[Mapping[str, Any]], raw_item: Mapping[str, Any]
) -> bool:
    available = {
        (str(response), bool(correct))
        for response, correct in zip(raw_item["outputs"], raw_item["correctness"])
    }
    return all((str(row["response"]), _correctness(row)) in available for row in pair)


def resample_strict_pairs(
    raw_records: Sequence[Mapping[str, Any]],
    original_records: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    seed: int = 2027,
) -> tuple[list[dict[str, Any]], dict[str, int | float | str]]:
    """Draw one correct and one incorrect response from each frozen response pool.

    Sampling is uniform over generation slots. The original response remains
    eligible, so duplicate slots retain their empirical multiplicity.
    """

    original_pairs = _group_original_pairs(original_records)
    by_question: dict[str, list[int]] = defaultdict(list)
    for pair_index, pair in enumerate(original_pairs):
        by_question[str(pair[0]["question"])].append(pair_index)

    used_pairs: set[int] = set()
    output: list[dict[str, Any]] = []
    summary: dict[str, int | float | str] = {
        "domain": domain,
        "seed": int(seed),
        "original_pairs": len(original_pairs),
        "aligned_original_pairs": 0,
        "eligible_r2_pairs": 0,
        "ineligible_after_validation": 0,
        "both_changed_pairs": 0,
        "correct_only_changed_pairs": 0,
        "incorrect_only_changed_pairs": 0,
        "unchanged_pairs": 0,
    }

    for raw_index, item in enumerate(raw_records):
        question = str(item["question"])
        matches = [
            pair_index
            for pair_index in by_question.get(question, [])
            if pair_index not in used_pairs
            and _pair_matches_pool(original_pairs[pair_index], item)
        ]
        if not matches:
            continue
        pair_index = matches[0]
        used_pairs.add(pair_index)
        summary["aligned_original_pairs"] = int(summary["aligned_original_pairs"]) + 1
        original_pair = original_pairs[pair_index]
        original_by_correct = {_correctness(row): row for row in original_pair}
        candidates = candidate_pool(item, domain)
        pools = {
            correctness: [row for row in candidates if row["correct"] is correctness]
            for correctness in (True, False)
        }
        if not pools[True] or not pools[False]:
            summary["ineligible_after_validation"] = (
                int(summary["ineligible_after_validation"]) + 1
            )
            continue

        selected: dict[bool, dict[str, Any]] = {}
        changed: dict[bool, bool] = {}
        for correctness, role in ((True, "correct"), (False, "incorrect")):
            rng = random.Random(stable_seed(seed, domain, raw_index, question, role))
            selected[correctness] = dict(rng.choice(pools[correctness]))
            changed[correctness] = (
                selected[correctness]["response"]
                != str(original_by_correct[correctness]["response"])
            )

        if changed[True] and changed[False]:
            change_scope = "both_changed"
        elif changed[True]:
            change_scope = "correct_only_changed"
        elif changed[False]:
            change_scope = "incorrect_only_changed"
        else:
            change_scope = "unchanged"
        summary[f"{change_scope}_pairs"] = int(summary[f"{change_scope}_pairs"]) + 1

        r2_pair_id = int(summary["eligible_r2_pairs"])
        for correctness, role in ((True, "correct"), (False, "incorrect")):
            choice = selected[correctness]
            row = {
                "question": question,
                "response": choice["response"],
                "answer": choice["answer"],
                "correct": correctness,
                "ground_truth": item.get("ground_truth"),
                "r2_pair_id": r2_pair_id,
                "r2_role": role,
                "r2_seed": int(seed),
                "raw_question_index": raw_index,
                "pass_index": int(choice["pass_index"]),
                "original_pair_index": pair_index,
                "r2_reused_original": not changed[correctness],
                "r2_change_scope": change_scope,
            }
            for key in ("dataset", "split"):
                if key in item:
                    row[key] = item[key]
            output.append(row)
        summary["eligible_r2_pairs"] = r2_pair_id + 1

    unmatched = set(range(len(original_pairs))).difference(used_pairs)
    if unmatched:
        raise ValueError(f"could not align {len(unmatched)} original strict pairs")
    summary["r2_rows"] = len(output)
    summary["eligibility_fraction_of_original_pairs"] = (
        int(summary["eligible_r2_pairs"]) / len(original_pairs) if original_pairs else 0.0
    )
    return output, summary


def multi_reference_question_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """Find exact prompt hashes associated with more than one reference label."""

    labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        question = str(row["question"])
        label = row.get("ground_truth", row.get("reference_answer", row.get("correct_answer")))
        if label is None:
            raise ValueError("Movies source row lacks a reference actor")
        labels[question].add(str(label).strip().casefold())
    return {
        hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        for question, values in labels.items()
        if len(values) > 1
    }


def filter_question_ids(
    rows: Sequence[Mapping[str, Any]], excluded_question_ids: set[str]
) -> list[dict[str, Any]]:
    """Remove rows whose exact-question hash belongs to an exclusion set."""

    output = []
    for row in rows:
        question_id = str(
            row.get(
                "question_id",
                hashlib.sha256(str(row["question"]).encode("utf-8")).hexdigest()[:16],
            )
        )
        if question_id not in excluded_question_ids:
            output.append(dict(row))
    return output
