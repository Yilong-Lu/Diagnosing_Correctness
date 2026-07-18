import json

import pytest

from metacog.robustness import (
    filter_question_ids,
    multi_reference_question_ids,
    resample_strict_pairs,
    summarize_scoring_rule,
)


def _math_pool():
    return {
        "question": "q",
        "ground_truth": "4",
        "outputs": [
            "Reasoning: old correct\nAnswer: 4",
            "Reasoning: new correct\nAnswer: 4",
            "Reasoning: old wrong\nAnswer: 5",
            "Reasoning: new wrong\nAnswer: 6",
        ],
        "answers": ["4", "4", "5", "6"],
        "correctness": [True, True, False, False],
    }


def _original_pair():
    return [
        {
            "question": "q",
            "response": "Reasoning: old correct\nAnswer: 4",
            "objective_correctness": 1,
        },
        {
            "question": "q",
            "response": "Reasoning: old wrong\nAnswer: 5",
            "objective_correctness": 0,
        },
    ]


def test_scoring_rule_audit_uses_paired_symmetric_selection():
    rows = [
        {"question": "q1", "correct": True, "old": 0.8, "full": 0.8},
        {"question": "q1", "correct": False, "old": 0.2, "full": 0.6},
        {"question": "q2", "correct": True, "old": 0.9, "full": 0.9},
        {"question": "q2", "correct": False, "old": 0.1, "full": 0.1},
    ]
    result = summarize_scoring_rule(
        rows, historical_key="old", full_key="full", threshold=0.7
    )
    assert result["label_flip_n"] == 1
    assert result["historical_complete_pair_n"] == 2
    assert result["full_complete_pair_n"] == 1
    assert result["complete_pair_membership_change_n"] == 1


def test_strict_pair_resampling_is_deterministic_and_keeps_original_eligible():
    first, first_summary = resample_strict_pairs(
        [_math_pool()], _original_pair(), domain="math", seed=2027
    )
    second, second_summary = resample_strict_pairs(
        [_math_pool()], _original_pair(), domain="math", seed=2027
    )
    assert first == second
    assert first_summary == second_summary
    assert len(first) == 2
    assert {row["correct"] for row in first} == {False, True}
    assert first_summary["eligible_r2_pairs"] == 1


def test_strict_pair_resampling_rejects_unaligned_original_pairs():
    original = _original_pair()
    original[1]["response"] = "not in the frozen pool"
    with pytest.raises(ValueError, match="could not align"):
        resample_strict_pairs([_math_pool()], original, domain="math", seed=2027)


def test_multireference_detection_and_filtering_are_question_hashed():
    source = [
        {"question": "ambiguous", "ground_truth": "Actor A"},
        {"question": "ambiguous", "ground_truth": "Actor B"},
        {"question": "unique", "ground_truth": "Actor C"},
    ]
    excluded = multi_reference_question_ids(source)
    assert len(excluded) == 1
    rows = [
        {"question": "ambiguous", "sample_id": 0},
        {"question": "unique", "sample_id": 1},
    ]
    assert filter_question_ids(rows, excluded) == [rows[1]]


def test_robustness_cli_writes_audit_summary(tmp_path):
    from metacog.cli.robustness import main

    source = tmp_path / "scores.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"question": "q", "correct": True, "old": 0.9, "full": 0.9},
                {"question": "q", "correct": False, "old": 0.1, "full": 0.1},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    main(
        [
            "score-audit",
            "--input",
            str(source),
            "--output",
            str(output),
            "--historical-key",
            "old",
            "--full-key",
            "full",
        ]
    )
    assert json.loads(output.read_text(encoding="utf-8"))["label_flip_n"] == 0
