"""Rebuild manuscript data figures and copy source-data tables."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..plotting import (
    plot_exp1,
    plot_exp2a,
    plot_exp2b,
    plot_null_controls,
    plot_ood,
    plot_threshold_distributions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/publication"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tables = args.input / "tables"
    all_layer = pd.read_csv(tables / "id_all_layer_question_cluster_intervals.csv")
    exp2a = pd.read_csv(tables / "exp2a_grouped_all_layers.csv")
    ood = pd.read_csv(tables / "ood_all_layer_transfer.csv")
    counterbalanced = pd.read_csv(tables / "ood_counterbalanced_all_layer_transfer.csv")
    nulls = pd.read_csv(tables / "window_null_controls.csv")
    figures = args.output / "figures"
    plot_exp1(all_layer, figures)
    plot_exp2a(exp2a, figures)
    plot_exp2b(all_layer, figures)
    plot_ood(ood, figures)
    plot_ood(counterbalanced, figures, stem="figureS_ood_counterbalanced")
    plot_null_controls(nulls, figures)
    plot_threshold_distributions(args.data, figures)

    output_tables = args.output / "tables"
    output_tables.mkdir(parents=True, exist_ok=True)
    for source in sorted(tables.glob("*.csv")):
        shutil.copyfile(source, output_tables / source.name)
    print(f"Wrote paper reproduction outputs to {args.output}")


if __name__ == "__main__":
    main()
