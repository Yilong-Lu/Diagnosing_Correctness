"""Deterministic descriptive summaries derived from all-layer result tables."""

from __future__ import annotations

import pandas as pd


def descriptive_exp2b_peaks(rows: pd.DataFrame) -> pd.DataFrame:
    """Select each model's layer with maximum mean delta across ID directions."""

    required = {"model", "layer", "source", "target", "cb_delta_auc"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Exp2B rows are missing columns: {sorted(missing)}")
    data = rows.copy()
    if "experiment" in data:
        data = data[data["experiment"].eq("Exp2B")]
    means = (
        data.groupby(["model", "layer"], as_index=False)["cb_delta_auc"]
        .mean()
        .rename(columns={"cb_delta_auc": "mean_across_directions_delta_auc"})
    )
    if means.empty:
        raise ValueError("no Exp2B rows were found")
    selected = means.loc[means.groupby("model")["mean_across_directions_delta_auc"].idxmax()]
    peak = data.merge(selected, on=["model", "layer"], how="inner")
    peak["mean_across_directions_delta_auc"] = peak[
        "mean_across_directions_delta_auc"
    ].round(12)
    return peak.sort_values(["model", "source", "target"]).reset_index(drop=True)


def ood_heterogeneity(rows: pd.DataFrame) -> pd.DataFrame:
    """Count direction-by-layer OOD effects by model and target benchmark."""

    required = {
        "model",
        "target",
        "delta_auc",
        "delta_auc_ci_low",
        "n_B",
        "n_C",
        "n_questions",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"OOD rows are missing columns: {sorted(missing)}")
    records = []
    for (model, target), group in rows.groupby(["model", "target"], sort=True):
        n_b = int(group["n_B"].iloc[0])
        n_c = int(group["n_C"].iloc[0])
        records.append(
            {
                "model": model,
                "target": target,
                "rows": int(len(group)),
                "mean_delta_auc": round(float(group["delta_auc"].mean()), 3),
                "positive_rows": int(group["delta_auc"].gt(0).sum()),
                "ci_above_zero_rows": int(group["delta_auc_ci_low"].gt(0).sum()),
                "n_B": n_b,
                "n_C": n_c,
                "n_BC": n_b + n_c,
                "n_questions": int(group["n_questions"].iloc[0]),
            }
        )
    return pd.DataFrame.from_records(records)
