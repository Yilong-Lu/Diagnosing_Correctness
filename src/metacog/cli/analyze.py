"""Run the manuscript-aligned experiments from activation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from ..experiments.controls import (
    null_window_control,
    nuisance_window_control,
    oc_only_window_control,
    question_fe_control,
    source_question_fe_control,
    token_matched_window_control,
)
from ..experiments.exp1 import run_exp1
from ..experiments.exp2a import run_exp2a, run_exp2a_window
from ..experiments.exp2b import (
    run_exp2b,
    run_exp2b_window,
    run_joint_source_target_bootstrap,
)
from ..io import load_activation_bundle, write_json


DEFAULT_SEEDS = {
    "exp1": 42,
    "exp2a": 20260707,
    "exp2a-window": 20260707,
    "exp2b": 42,
    "exp2b-window": 20260702,
    "joint-exp2b": 20260712,
    "oc-only": 20260707,
    "question-fe": 20260707,
    "source-question-fe": 20260715,
    "nuisance": 20260707,
    "token-match": 42,
    "null-controls": 42,
    "ood": 20260702,
    "ood-window": 20260702,
}


def resolve_seed(experiment: str, requested: int | None, target_domain: str | None) -> int:
    if requested is not None:
        return int(requested)
    if experiment == "nuisance" and target_domain in {"mmlu", "truthfulqa_binary"}:
        return 20260705
    return DEFAULT_SEEDS[experiment]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("analysis produced no rows")
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        choices=[
            "exp1",
            "exp2a",
            "exp2a-window",
            "exp2b",
            "exp2b-window",
            "joint-exp2b",
            "oc-only",
            "question-fe",
            "source-question-fe",
            "nuisance",
            "token-match",
            "null-controls",
            "ood",
            "ood-window",
        ],
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--window-start", type=float, default=0.40)
    parser.add_argument("--window-end", type=float, default=0.80)
    parser.add_argument("--nuisance-csv", type=Path)
    parser.add_argument("--nuisance-columns", nargs="+", default=["mean_answer_logprob", "token_count"])
    parser.add_argument("--matching-bins", nargs="+", type=int, default=[20, 10, 5, 2, 1])
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source = load_activation_bundle(args.source)
    target = load_activation_bundle(args.target) if args.target else None
    seed = resolve_seed(
        args.experiment,
        args.seed,
        target.domain if target is not None else None,
    )

    if args.experiment == "exp2a":
        write_rows(
            args.output,
            run_exp2a(
                source,
                n_splits=args.folds,
                bootstrap_repetitions=args.bootstrap,
                seed=seed,
            ),
        )
        return
    if args.experiment == "exp2a-window":
        write_json(
            args.output,
            run_exp2a_window(
                source,
                n_splits=args.folds,
                repetitions=args.bootstrap,
                seed=seed,
                window_start=args.window_start,
                window_end=args.window_end,
            ),
        )
        return
    if target is None:
        raise ValueError(f"{args.experiment} requires --target")
    if args.experiment == "question-fe":
        result = question_fe_control(
            source,
            target,
            repetitions=args.bootstrap,
            seed=seed,
            window_start=args.window_start,
            window_end=args.window_end,
        )
        write_json(args.output, result)
        return
    if args.experiment == "source-question-fe":
        result = source_question_fe_control(
            source,
            target,
            repetitions=args.bootstrap,
            seed=seed,
            window_start=args.window_start,
            window_end=args.window_end,
        )
        write_json(args.output, result)
        return
    if args.experiment == "nuisance":
        if args.nuisance_csv is None:
            raise ValueError("nuisance analysis requires --nuisance-csv")
        import numpy as np
        import pandas as pd

        sidecar = pd.read_csv(args.nuisance_csv).sort_values("sample_id")
        if sidecar["sample_id"].tolist() != target.sample_ids.tolist():
            raise ValueError("nuisance sidecar sample IDs do not align with target activations")
        nuisance = sidecar[args.nuisance_columns].to_numpy(dtype=np.float64)
        result = nuisance_window_control(
            source,
            target,
            nuisance,
            repetitions=args.bootstrap,
            seed=seed,
            window_start=args.window_start,
            window_end=args.window_end,
        )
        write_json(args.output, result)
        return
    if args.experiment == "token-match":
        write_json(
            args.output,
            token_matched_window_control(
                source,
                target,
                repetitions=args.bootstrap,
                seed=seed,
                window_start=args.window_start,
                window_end=args.window_end,
                candidate_bins=tuple(args.matching_bins),
            ),
        )
        return
    if args.experiment in {"exp2b-window", "ood-window"}:
        write_json(
            args.output,
            run_exp2b_window(
                source,
                target,
                repetitions=args.bootstrap,
                seed=seed,
                window_start=args.window_start,
                window_end=args.window_end,
            ),
        )
        return
    if args.experiment == "null-controls":
        write_json(
            args.output,
            null_window_control(
                source,
                target,
                repetitions=args.bootstrap,
                seed=seed,
                window_start=args.window_start,
                window_end=args.window_end,
            ),
        )
        return
    if args.experiment == "exp1":
        rows = run_exp1(
            source, target, bootstrap_repetitions=args.bootstrap, seed=seed
        )
    elif args.experiment in {"exp2b", "ood"}:
        rows = run_exp2b(
            source, target, bootstrap_repetitions=args.bootstrap, seed=seed
        )
    elif args.experiment == "oc-only":
        write_json(
            args.output,
            oc_only_window_control(
                source,
                target,
                repetitions=args.bootstrap,
                seed=seed,
                window_start=args.window_start,
                window_end=args.window_end,
            ),
        )
        return
    else:
        result = run_joint_source_target_bootstrap(
            source,
            target,
            repetitions=args.bootstrap,
            seed=seed,
            window_start=args.window_start,
            window_end=args.window_end,
        )
        result.update(
            {
                "model": source.model,
                "source": source.domain,
                "target": target.domain,
            }
        )
        write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
