"""Build deterministic descriptive summaries from all-layer result CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..reporting import descriptive_exp2b_peaks, ood_heterogeneity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["exp2b-peaks", "ood-heterogeneity"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = pd.read_csv(args.input)
    if args.kind == "exp2b-peaks":
        result = descriptive_exp2b_peaks(rows)
    else:
        result = ood_heterogeneity(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
