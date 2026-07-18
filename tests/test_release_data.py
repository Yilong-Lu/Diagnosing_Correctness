import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_data_manifest_checksums_and_forbidden_targets():
    rows = list(csv.DictReader((ROOT / "data/manifest.csv").open(encoding="utf-8")))
    assert rows
    assert not any("mc1" in row["path"].lower() for row in rows)
    for row in rows:
        path = ROOT / row["path"]
        assert path.stat().st_size == int(row["bytes"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_released_id_strict_counts_match_publication_table():
    attrition = pd.read_csv(ROOT / "artifacts/publication/tables/id_attrition.csv")
    for row in attrition.itertuples(index=False):
        path = ROOT / f"data/processed/id/{row.model}/{row.domain}/strict_pairs.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == row.high_confidence_rows
        assert [record["sample_id"] for record in records] == list(range(len(records)))


def test_released_r2_records_match_reported_sample_flow():
    sample_flow = pd.read_csv(
        ROOT / "artifacts/publication/tables/qwen25_7b_r2_sample_flow.csv"
    )
    for row in sample_flow.itertuples(index=False):
        root = ROOT / f"data/processed/robustness/qwen25_7b_r2/{row.domain}"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        strict = (root / "strict_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        assert summary["all_pairs"] == row.original_pairs
        assert summary["strict_pairs"] == row.strict_pairs_tau07
        assert len(strict) == 2 * row.strict_pairs_tau07


def test_movies_multireference_hash_release_has_reported_support():
    hashes = (
        ROOT / "data/processed/robustness/movies_multireference_question_ids.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert len(hashes) == 159
    assert len(set(hashes)) == 159
    assert all(len(value) == 16 for value in hashes)
