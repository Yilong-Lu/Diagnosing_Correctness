import json

import numpy as np

from metacog.cli.prepare import normalize_id_pairs, normalize_ood_candidates, rethreshold_id_pairs


def test_prepare_preserves_pairs_and_applies_pairwise_threshold(tmp_path):
    source = tmp_path / "pairs.json"
    source.write_text(
        json.dumps(
            [
                {"question": "q1", "response": "a", "correct": True, "p_judgement": 0.8},
                {"question": "q1", "response": "b", "correct": False, "p_judgement": 0.2},
                {"question": "q2", "response": "c", "correct": True, "p_judgement": 0.6},
                {"question": "q2", "response": "d", "correct": False, "p_judgement": 0.1},
            ]
        ),
        encoding="utf-8",
    )
    summary = normalize_id_pairs(
        source, tmp_path / "out", model="toy", domain="math", threshold=0.7
    )
    assert summary == {
        "all_rows": 4,
        "all_pairs": 2,
        "strict_rows": 2,
        "strict_pairs": 1,
        "unique_question_clusters": 1,
    }
    strict = [json.loads(line) for line in (tmp_path / "out" / "strict_pairs.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in strict] == [0, 1]
    assert [row["source_sample_id"] for row in strict] == [0, 1]


def test_prepare_ood_keeps_only_confident_conflicts_for_activation(tmp_path):
    source = tmp_path / "ood.jsonl"
    rows = [
        {"question_id": "q1", "question": "q", "response": "A", "correct": True, "p_judgement": 0.1},
        {"question_id": "q1", "question": "q", "response": "B", "correct": False, "p_judgement": 0.2},
        {"question_id": "q2", "question": "r", "response": "A", "correct": False, "p_judgement": 0.8},
        {"question_id": "q2", "question": "r", "response": "B", "correct": True, "p_judgement": 0.6},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = normalize_ood_candidates(
        source,
        tmp_path / "out_ood",
        model="toy",
        domain="mmlu",
        threshold=0.7,
    )
    assert summary["strict_rows"] == 3
    assert summary["strict_conflict_rows"] == 2
    assert summary["strict_conflict_questions"] == 2
    conflicts = [
        json.loads(line)
        for line in (tmp_path / "out_ood" / "strict_conflicts.jsonl").read_text().splitlines()
    ]
    assert [row["sample_id"] for row in conflicts] == [0, 1]
    assert [row["source_sample_id"] for row in conflicts] == [0, 2]


def test_rethresholds_released_all_pairs_without_original_json(tmp_path):
    source = tmp_path / "all_pairs.jsonl"
    rows = [
        {"sample_id": 0, "pair_id": "p1", "p_self_judgement": 0.6},
        {"sample_id": 1, "pair_id": "p1", "p_self_judgement": 0.4},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = rethreshold_id_pairs(source, tmp_path / "strict.jsonl", threshold=0.5)
    assert summary["strict_rows"] == 2
    strict = [json.loads(line) for line in (tmp_path / "strict.jsonl").read_text().splitlines()]
    assert [row["self_judgement"] for row in strict] == [1, 0]


def test_rethreshold_is_row_level_unless_pairwise_is_requested(tmp_path):
    source = tmp_path / "all_pairs.jsonl"
    rows = [
        {"sample_id": 0, "pair_id": "p1", "p_self_judgement": 0.8},
        {"sample_id": 1, "pair_id": "p1", "p_self_judgement": 0.6},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    row_summary = rethreshold_id_pairs(source, tmp_path / "row.jsonl", threshold=0.7)
    pair_summary = rethreshold_id_pairs(
        source, tmp_path / "pair.jsonl", threshold=0.7, pairwise=True
    )
    assert row_summary["strict_rows"] == 1
    assert pair_summary["strict_rows"] == 0


def test_counterbalanced_raw_log_odds_are_sigmoided(tmp_path):
    source = tmp_path / "balanced.jsonl"
    source.write_text(
        json.dumps(
            {
                "question_id": "q",
                "question": "q",
                "response": "A",
                "correct": False,
                "correct_log_odds_balanced": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    normalize_ood_candidates(
        source, tmp_path / "balanced", model="toy", domain="mmlu", threshold=0.7
    )
    row = json.loads((tmp_path / "balanced" / "all_candidates.jsonl").read_text())
    assert np.isclose(row["p_self_judgement"], 1.0 / (1.0 + np.exp(-2.0)))
