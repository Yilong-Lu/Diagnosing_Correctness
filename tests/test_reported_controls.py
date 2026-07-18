import numpy as np

from metacog.experiments.controls import (
    null_window_control,
    oc_only_window_control,
    token_matched_window_control,
)
from metacog.cli.analyze import build_parser, resolve_seed


def test_analysis_cli_exposes_reported_controls():
    parser = build_parser()
    choices = parser._actions[1].choices
    assert {"oc-only", "token-match", "null-controls"}.issubset(set(choices))
    args = parser.parse_args(
        [
            "token-match",
            "--source",
            "s",
            "--target",
            "t",
            "--output",
            "o",
            "--matching-bins",
            "10",
        ]
    )
    assert args.matching_bins == [10]


def test_nuisance_default_seed_tracks_target_family():
    assert resolve_seed("nuisance", None, "movies") == 20260707
    assert resolve_seed("nuisance", None, "mmlu") == 20260705
    assert resolve_seed("nuisance", 19, "mmlu") == 19


def test_oc_only_control_reports_cluster_interval(synthetic_bundle_factory):
    source = synthetic_bundle_factory("toy", "math", 0.0)
    target = synthetic_bundle_factory("toy", "movies", 0.1)
    result = oc_only_window_control(source, target, repetitions=20, seed=9)
    assert result["n_boot_valid"] == 20
    assert result["auc_ci_low"] <= result["auc_c_above_b"] <= result["auc_ci_high"]


def test_token_matched_control_reports_effect(synthetic_bundle_factory):
    source = synthetic_bundle_factory("toy", "math", 0.0)
    target = synthetic_bundle_factory("toy", "movies", 0.1)
    for index, record in enumerate(target.records):
        record["token_count"] = 10 + index % 4
    result = token_matched_window_control(source, target, repetitions=20, seed=10)
    assert result["matched_n"] > 0
    assert np.isfinite(result["matched_delta_cb"])
    assert result["n_boot_valid"] > 0


def test_null_control_is_seed_reproducible(synthetic_bundle_factory):
    source = synthetic_bundle_factory("toy", "math", 0.0)
    target = synthetic_bundle_factory("toy", "movies", 0.1)
    first = null_window_control(source, target, repetitions=10, seed=11)
    second = null_window_control(source, target, repetitions=10, seed=11)
    assert first == second
    assert set(first) >= {"observed_delta_cb", "label_shuffle_ci_low", "random_ci_high"}
