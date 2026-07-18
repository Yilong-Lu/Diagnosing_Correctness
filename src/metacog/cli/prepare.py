"""Normalize frozen response pairs into the release schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..io import stable_hash, write_json, write_jsonl
from ..thresholding import apply_symmetric_confidence, paired_confidence_mask


def _reindex_for_activation(rows: list[dict]) -> list[dict]:
    reindexed = []
    for sample_id, row in enumerate(rows):
        item = dict(row)
        item["source_sample_id"] = int(row["sample_id"])
        item["sample_id"] = sample_id
        reindexed.append(item)
    return reindexed


def normalize_id_pairs(
    input_path: Path,
    output_dir: Path,
    *,
    model: str,
    domain: str,
    threshold: float,
) -> dict[str, int]:
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) % 2:
        raise ValueError("ID judged-pair input must be an even-length JSON list")

    normalized = []
    for index, row in enumerate(rows):
        pair_index = index // 2
        pair = rows[2 * pair_index : 2 * pair_index + 2]
        if len(pair) != 2 or pair[0]["question"] != pair[1]["question"]:
            raise ValueError(f"rows {2 * pair_index}:{2 * pair_index + 2} are not a pair")
        if {bool(item["correct"]) for item in pair} != {False, True}:
            raise ValueError(f"pair {pair_index} does not contain one correct and one incorrect response")
        question = str(row["question"])
        normalized.append(
            {
                "sample_id": index,
                "model": model,
                "domain": domain,
                "pair_id": f"{domain}_pair_{pair_index:06d}",
                "question_id": stable_hash(question),
                "question": question,
                "response": str(row.get("response", row.get("answer", ""))),
                "answer": row.get("answer", row.get("response", "")),
                "reference_answer": row.get("ground_truth", row.get("correct_answer")),
                "objective_correctness": int(bool(row["correct"])),
                "p_self_judgement": float(row["p_judgement"]),
                "source_split": row.get("split"),
            }
        )

    probabilities = [row["p_self_judgement"] for row in normalized]
    pair_ids = [row["pair_id"] for row in normalized]
    thresholded = apply_symmetric_confidence(probabilities, threshold)
    pair_keep = paired_confidence_mask(probabilities, pair_ids, threshold)
    for row, label, keep in zip(normalized, thresholded.labels, pair_keep):
        row["self_judgement"] = int(label)
        row["threshold_keep"] = bool(keep)
        row["threshold"] = float(threshold)
        row["threshold_mode"] = "symmetric_confident"

    strict = _reindex_for_activation([row for row in normalized if row["threshold_keep"]])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "all_pairs.jsonl", normalized)
    write_jsonl(output_dir / "strict_pairs.jsonl", strict)
    summary = {
        "all_rows": len(normalized),
        "all_pairs": len(normalized) // 2,
        "strict_rows": len(strict),
        "strict_pairs": len(strict) // 2,
        "unique_question_clusters": len({row["question_id"] for row in strict}),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _read_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("expected a JSON list or JSON Lines input")
    return payload


def rethreshold_id_pairs(
    input_path: Path,
    output_path: Path,
    *,
    threshold: float,
    pairwise: bool = False,
) -> dict[str, int]:
    """Apply a new symmetric-confidence threshold to released all-pairs records."""

    rows = _read_json_or_jsonl(input_path)
    probabilities = [float(row["p_self_judgement"]) for row in rows]
    pair_ids = [str(row["pair_id"]) for row in rows]
    labels = apply_symmetric_confidence(probabilities, threshold)
    keep = (
        paired_confidence_mask(probabilities, pair_ids, threshold)
        if pairwise
        else labels.keep
    )
    selected = []
    for row, label, retained in zip(rows, labels.labels, keep):
        if not retained:
            continue
        item = dict(row)
        item["self_judgement"] = int(label)
        item["threshold_keep"] = True
        item["threshold"] = float(threshold)
        item["threshold_mode"] = "symmetric_confident"
        selected.append(item)
    strict = _reindex_for_activation(selected)
    write_jsonl(output_path, strict)
    return {
        "all_rows": len(rows),
        "strict_rows": len(strict),
        "strict_pairs": len(strict) // 2,
        "filter_level": "pair" if pairwise else "row",
        "unique_question_clusters": len({row["question_id"] for row in strict if "question_id" in row}),
    }


def normalize_ood_candidates(
    input_path: Path,
    output_dir: Path,
    *,
    model: str,
    domain: str,
    threshold: float,
) -> dict[str, int]:
    rows = _read_json_or_jsonl(input_path)
    normalized = []
    for index, row in enumerate(rows):
        if "p_judgement" in row:
            probability = float(row["p_judgement"])
        elif "p_correct_balanced_probability" in row:
            probability = float(row["p_correct_balanced_probability"])
        elif "correct_log_odds_balanced" in row:
            log_odds = float(row["correct_log_odds_balanced"])
            probability = float(1.0 / (1.0 + np.exp(-log_odds)))
        elif "p_correct_balanced_log_odds" in row:
            # Historical counterbalanced artifacts used this misleading key for
            # an already-sigmoided probability. It remains a compatibility path.
            probability = float(row["p_correct_balanced_log_odds"])
        else:
            raise ValueError("OOD rows require p_judgement or p_correct_balanced_log_odds")
        question = str(row["question"])
        question_id = str(row.get("question_id", stable_hash(question)))
        normalized.append(
            {
                "sample_id": index,
                "model": model,
                "domain": domain,
                "pair_id": question_id,
                "question_id": question_id,
                "question": question,
                "prompt": str(row.get("prompt", question)),
                "response": str(row.get("response", row.get("answer", ""))),
                "answer": str(row.get("answer", row.get("response", ""))),
                "reference_answer": row.get("ground_truth", row.get("correct_answer")),
                "objective_correctness": int(bool(row["correct"])),
                "p_self_judgement": probability,
                "candidate_index": row.get("candidate_index"),
                "option_label": row.get("option_label"),
                "subject": row.get("subject"),
                "category": row.get("category"),
            }
        )
    result = apply_symmetric_confidence(
        [row["p_self_judgement"] for row in normalized], threshold
    )
    for row, label, keep in zip(normalized, result.labels, result.keep):
        row["self_judgement"] = int(label)
        row["threshold_keep"] = bool(keep)
        row["threshold"] = float(threshold)
        row["threshold_mode"] = "symmetric_confident"
    strict_source = [row for row in normalized if row["threshold_keep"]]
    conflicts_source = [
        row
        for row in strict_source
        if row["objective_correctness"] != row["self_judgement"]
    ]
    strict = _reindex_for_activation(strict_source)
    conflicts = _reindex_for_activation(conflicts_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "all_candidates.jsonl", normalized)
    write_jsonl(output_dir / "strict_candidates.jsonl", strict)
    write_jsonl(output_dir / "strict_conflicts.jsonl", conflicts)
    summary = {
        "all_rows": len(normalized),
        "strict_rows": len(strict),
        "strict_conflict_rows": len(conflicts),
        "strict_conflict_questions": len({row["question_id"] for row in conflicts}),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=["id-pairs", "ood-candidates", "rethreshold-id"],
        default="id-pairs",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--domain",
        choices=["math", "movies", "mmlu", "truthfulqa_binary"],
        required=True,
    )
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Require both members of an ID pair to pass the confidence threshold.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.kind == "rethreshold-id":
        summary = rethreshold_id_pairs(
            args.input,
            args.output,
            threshold=args.threshold,
            pairwise=args.pairwise,
        )
    elif args.kind == "id-pairs":
        if args.domain not in {"math", "movies"}:
            raise ValueError("id-pairs supports only math and movies")
        summary = normalize_id_pairs(
            args.input,
            args.output,
            model=args.model,
            domain=args.domain,
            threshold=args.threshold,
        )
    else:
        if args.domain not in {"mmlu", "truthfulqa_binary"}:
            raise ValueError("ood-candidates supports only mmlu and truthfulqa_binary")
        summary = normalize_ood_candidates(
            args.input,
            args.output,
            model=args.model,
            domain=args.domain,
            threshold=args.threshold,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
