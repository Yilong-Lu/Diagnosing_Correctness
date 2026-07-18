import numpy as np

from metacog.experiments.exp1 import run_exp1
from metacog.experiments.exp2a import run_exp2a, run_exp2a_window
from metacog.experiments.exp2b import (
    run_exp2b,
    run_exp2b_window,
    run_joint_source_target_bootstrap,
)
from metacog.experiments.controls import (
    counterbalanced_probability,
    paired_difference_design,
    question_fe_control,
    source_question_fe_control,
    weighted_paired_fe_directions,
)


def test_layerwise_experiments_return_all_layers(synthetic_bundle_factory):
    source = synthetic_bundle_factory(domain="math")
    target = synthetic_bundle_factory(domain="movies", shift=0.3)
    assert len(run_exp1(source, target)) == 3
    assert len(run_exp2a(source, n_splits=3)) == 3
    rows = run_exp2b(source, target)
    assert len(rows) == 3
    assert all(np.isclose(row["meta_to_sj_auc"], 1.0) for row in rows)


def test_joint_bootstrap_is_seeded_and_refits_source(synthetic_bundle_factory):
    source = synthetic_bundle_factory(domain="math")
    target = synthetic_bundle_factory(domain="movies", shift=0.2)
    first = run_joint_source_target_bootstrap(
        source, target, repetitions=20, seed=9, window_start=0.0, window_end=1.0
    )
    second = run_joint_source_target_bootstrap(
        source, target, repetitions=20, seed=9, window_start=0.0, window_end=1.0
    )
    assert first == second
    assert first["n_boot_requested"] == 20
    assert first["n_layers"] == 3


def test_exp2a_window_uses_grouped_oof_scores(synthetic_bundle_factory):
    bundle = synthetic_bundle_factory(domain="math")
    result = run_exp2a_window(
        bundle,
        n_splits=3,
        repetitions=20,
        seed=11,
        window_start=0.0,
        window_end=1.0,
    )
    assert result["n_layers"] == 3
    assert result["n_splits"] == 3
    assert result["n_boot_requested"] == 20
    assert result["n_boot_valid"] > 0
    assert {"meta_to_sj_auc", "truth_to_oc_auc", "delta_cb"}.issubset(result)


def test_exp2b_fixed_source_window_is_seeded(synthetic_bundle_factory):
    source = synthetic_bundle_factory(domain="math")
    target = synthetic_bundle_factory(domain="movies", shift=0.2)
    first = run_exp2b_window(
        source,
        target,
        repetitions=20,
        seed=13,
        window_start=0.0,
        window_end=1.0,
    )
    second = run_exp2b_window(
        source,
        target,
        repetitions=20,
        seed=13,
        window_start=0.0,
        window_end=1.0,
    )
    assert first == second
    assert first["n_layers"] == 3
    assert first["n_boot_valid"] > 0


def test_question_fixed_effect_control_runs_on_all_cells(synthetic_bundle_factory):
    source = synthetic_bundle_factory(domain="math")
    target = synthetic_bundle_factory(domain="movies", shift=0.2)
    result = question_fe_control(
        source, target, repetitions=10, seed=4, window_start=0.0, window_end=1.0
    )
    assert result["n_boot_valid"] == 10
    assert result["n_questions"] == 12
    assert set(result["coefficients"]) == {
        "meta_oc",
        "meta_sj",
        "meta_interaction",
        "truth_oc",
        "truth_sj",
        "truth_interaction",
    }
    assert all(
        "ci_low" in value and "ci_high" in value
        for value in result["coefficients"].values()
    )


def test_question_fixed_effect_control_reports_within_component_specificity(
    synthetic_bundle_factory,
):
    source = synthetic_bundle_factory(domain="math")
    target = synthetic_bundle_factory(domain="movies", shift=0.2)
    result = question_fe_control(
        source, target, repetitions=10, seed=4, window_start=0.0, window_end=1.0
    )

    specificity = result["specificity"]
    coefficients = result["coefficients"]
    assert set(specificity) == {"meta_sj_minus_oc", "truth_oc_minus_sj"}
    assert np.isclose(
        specificity["meta_sj_minus_oc"]["estimate"],
        coefficients["meta_sj"]["estimate"] - coefficients["meta_oc"]["estimate"],
    )
    assert np.isclose(
        specificity["truth_oc_minus_sj"]["estimate"],
        coefficients["truth_oc"]["estimate"] - coefficients["truth_sj"]["estimate"],
    )
    assert all(
        {"ci_low", "ci_high", "p_le_zero", "p_ge_zero"}.issubset(value)
        for value in specificity.values()
    )


def test_counterbalanced_probability_is_mapping_invariant():
    probability = counterbalanced_probability(-0.1, -2.1, -3.0, -0.2)
    assert probability > 0.8


def test_paired_source_fe_removes_additive_question_offsets():
    sj_correct = np.asarray([1, 1, 0, 0], dtype=np.int8)
    sj_incorrect = np.asarray([1, 0, 1, 0], dtype=np.int8)
    design = np.column_stack([
        np.full(4, 2.0),
        2.0 * sj_correct - 1.0 - (2.0 * sj_incorrect - 1.0),
        2.0 * sj_correct - 1.0 + (2.0 * sj_incorrect - 1.0),
    ])
    coefficients = np.asarray(
        [[1.5, -0.5], [0.25, 2.0], [-1.0, 0.75]], dtype=np.float32
    )
    differences = design @ coefficients
    offsets = np.asarray(
        [[10.0, -4.0], [-7.0, 3.0], [20.0, 11.0], [2.0, -13.0]],
        dtype=np.float32,
    )
    activations = np.empty((8, 2), dtype=np.float32)
    activations[0::2] = offsets + 0.5 * differences
    activations[1::2] = offsets - 0.5 * differences
    oc = np.tile(np.asarray([1, 0], dtype=np.int8), 4)
    sj = np.column_stack([sj_correct, sj_incorrect]).reshape(-1)
    groups = np.repeat(np.asarray(["q0", "q1", "q2", "q3"]), 2)

    paired = paired_difference_design(activations, oc, sj, groups)
    meta, truth, interaction, valid, ranks = weighted_paired_fe_directions(
        paired.differences,
        paired.design,
        np.ones((1, paired.n_pairs), dtype=np.float32),
    )

    assert paired.pattern_counts == {"AC": 1, "AD": 1, "BC": 1, "BD": 1}
    assert valid.tolist() == [True]
    assert ranks.tolist() == [3]
    assert np.allclose(truth[0], coefficients[0])
    assert np.allclose(meta[0], coefficients[1])
    assert np.allclose(interaction[0], coefficients[2])


def test_source_question_fe_control_runs_joint_bootstrap(synthetic_bundle_factory):
    source = synthetic_bundle_factory(domain="math")
    target = synthetic_bundle_factory(domain="movies", shift=0.2)

    result = source_question_fe_control(
        source,
        target,
        repetitions=20,
        seed=17,
        window_start=0.0,
        window_end=1.0,
    )

    assert result["source_design_rank"] == 3
    assert result["source_pair_AC"] == 3
    assert result["source_pair_AD"] == 3
    assert result["source_pair_BC"] == 3
    assert result["source_pair_BD"] == 3
    assert result["n_boot_valid"] == 20
    assert np.isfinite(result["delta_cb"])


def test_analysis_cli_exposes_source_question_fe_control(tmp_path):
    from metacog.cli.analyze import build_parser

    args = build_parser().parse_args([
        "source-question-fe",
        "--source", str(tmp_path / "source"),
        "--target", str(tmp_path / "target"),
        "--output", str(tmp_path / "result.json"),
    ])

    assert args.experiment == "source-question-fe"
