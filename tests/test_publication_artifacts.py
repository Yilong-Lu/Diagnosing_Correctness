from pathlib import Path

import pandas as pd


def test_publication_tables_cover_primary_evidence():
    root = Path("artifacts/publication/tables")
    all_layer = pd.read_csv(root / "id_all_layer_question_cluster_intervals.csv")
    exp2a = pd.read_csv(root / "exp2a_grouped_all_layers.csv")
    joint = pd.read_csv(root / "exp2b_joint_source_target_window.csv")
    source_fe = pd.read_csv(root / "source_question_fe_window.csv")
    target_fe = pd.read_csv(root / "question_fe_contrast.csv")
    ood = pd.read_csv(root / "ood_all_layer_transfer.csv")
    counterbalanced = pd.read_csv(root / "ood_counterbalanced_all_layer_transfer.csv")
    scoring = pd.read_csv(root / "judgement_scoring_rule_audit.csv")
    r2 = pd.read_csv(root / "qwen25_7b_r2_exp2b.csv")
    multireference = pd.read_csv(root / "movies_multireference_exp2b.csv")
    assert set(all_layer["experiment"]) == {"Exp1", "Exp2B"}
    exp1_cross_domain = all_layer[
        (all_layer["experiment"] == "Exp1") & (all_layer["source"] != all_layer["target"])
    ]
    assert len(exp1_cross_domain) == 280
    assert set(exp2a["experiment"]) == {"Exp2A_GroupKFold"}
    assert len(exp2a) == 280
    assert len(joint) == 8
    assert len(source_fe) == 8
    assert len(ood) == 560
    assert len(counterbalanced) == 560
    assert (joint["n_boot"] == 1000).all()
    assert (source_fe["n_boot_valid"] == 1000).all()
    assert (source_fe["source_design_rank"] == 3).all()
    assert (source_fe["delta_ci_low"] > 0).all()
    assert (target_fe["meta_sj_minus_oc_ci_low"] > 0).all()
    assert (target_fe["truth_oc_minus_sj_ci_high"] < 0).all()
    assert "llama31_8b" in set(joint["model"])
    assert "truthfulqa_binary" in set(ood["target"])
    assert "llama31_8b" in set(counterbalanced["model"])
    assert "truthfulqa_binary" in set(counterbalanced["target"])
    assert len(scoring) == 8
    assert scoring["top4_vs_full_max_abs_error"].max() < 1.31e-6
    assert scoring["top4_vs_full_label_flip_n"].sum() == 0
    assert scoring["top4_vs_full_complete_pair_membership_change_n"].sum() == 0
    assert len(r2) == 2
    assert (r2["r2_window_delta_ci_low"] > 0).all()
    assert r2["curve_correlation"].min() > 0.99
    assert len(multireference) == 8
    assert (multireference["filtered_delta_ci_low"] > 0).all()
    assert multireference["curve_correlation"].min() > 0.999
