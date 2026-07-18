"""Run the data-construction robustness checks reported in the supplement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..io import read_jsonl, write_json, write_jsonl
from ..robustness import (
    filter_question_ids,
    multi_reference_question_ids,
    resample_strict_pairs,
    summarize_scoring_rule,
)


def _read_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("expected a JSON list or JSON Lines input")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scoring = subparsers.add_parser(
        "score-audit", help="compare historical and full Yes/No scores at fixed logits"
    )
    scoring.add_argument("--input", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--historical-key", default="p_judgement_reconstructed")
    scoring.add_argument("--full-key", default="p_judgement_full")
    scoring.add_argument("--threshold", type=float, default=0.7)

    resample = subparsers.add_parser(
        "resample-pairs", help="repeat the strict-pair draw from frozen response pools"
    )
    resample.add_argument("--pool", type=Path, required=True)
    resample.add_argument("--original", type=Path, required=True)
    resample.add_argument("--output", type=Path, required=True)
    resample.add_argument("--summary", type=Path, required=True)
    resample.add_argument("--domain", choices=["math", "movies"], required=True)
    resample.add_argument("--seed", type=int, default=2027)

    multireference = subparsers.add_parser(
        "filter-multireference",
        help="remove records whose Movies prompt has multiple reference actors",
    )
    multireference.add_argument("--source", type=Path)
    multireference.add_argument("--question-ids", type=Path)
    multireference.add_argument("--input", type=Path, required=True)
    multireference.add_argument("--output", type=Path, required=True)
    multireference.add_argument("--ids-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "score-audit":
        result = summarize_scoring_rule(
            _read_rows(args.input),
            historical_key=args.historical_key,
            full_key=args.full_key,
            threshold=args.threshold,
        )
        write_json(args.output, result)
    elif args.command == "resample-pairs":
        rows, summary = resample_strict_pairs(
            _read_rows(args.pool),
            _read_rows(args.original),
            domain=args.domain,
            seed=args.seed,
        )
        write_jsonl(args.output, rows)
        write_json(args.summary, summary)
    else:
        if bool(args.source) == bool(args.question_ids):
            raise ValueError("provide exactly one of --source or --question-ids")
        if args.source:
            question_ids = multi_reference_question_ids(_read_rows(args.source))
        else:
            question_ids = {
                line.strip()
                for line in args.question_ids.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        rows = _read_rows(args.input)
        filtered = filter_question_ids(rows, question_ids)
        write_jsonl(args.output, filtered)
        if args.ids_output:
            args.ids_output.parent.mkdir(parents=True, exist_ok=True)
            args.ids_output.write_text("\n".join(sorted(question_ids)) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "excluded_question_ids": len(question_ids),
                    "input_rows": len(rows),
                    "output_rows": len(filtered),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    payload = result if args.command == "score-audit" else summary
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
