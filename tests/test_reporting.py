import pandas as pd

from metacog.reporting import descriptive_exp2b_peaks, ood_heterogeneity


def test_descriptive_peak_uses_mean_across_directions():
    rows = pd.DataFrame(
        [
            {"experiment": "Exp2B", "model": "m", "layer": 1, "source": "math", "target": "movies", "cb_delta_auc": 0.2},
            {"experiment": "Exp2B", "model": "m", "layer": 1, "source": "movies", "target": "math", "cb_delta_auc": 0.4},
            {"experiment": "Exp2B", "model": "m", "layer": 2, "source": "math", "target": "movies", "cb_delta_auc": 0.8},
            {"experiment": "Exp2B", "model": "m", "layer": 2, "source": "movies", "target": "math", "cb_delta_auc": -0.4},
        ]
    )
    peak = descriptive_exp2b_peaks(rows)
    assert peak["layer"].tolist() == [1, 1]
    assert peak["mean_across_directions_delta_auc"].tolist() == [0.3, 0.3]


def test_ood_heterogeneity_counts_rows_and_support():
    rows = pd.DataFrame(
        [
            {"model": "m", "target": "mmlu", "delta_auc": 0.2, "delta_auc_ci_low": 0.1, "n_B": 3, "n_C": 4, "n_questions": 5},
            {"model": "m", "target": "mmlu", "delta_auc": -0.1, "delta_auc_ci_low": -0.2, "n_B": 3, "n_C": 4, "n_questions": 5},
        ]
    )
    summary = ood_heterogeneity(rows)
    row = summary.iloc[0]
    assert row["rows"] == 2
    assert row["positive_rows"] == 1
    assert row["ci_above_zero_rows"] == 1
    assert row["n_BC"] == 7
