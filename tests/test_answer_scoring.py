import numpy as np

from metacog.scoring import (
    align_response_tokens,
    forced_choice_row,
    token_ids_for_choice,
)
from metacog.cli.score import build_parser


class ToyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        mapping = {
            "A": [10],
            " A": [11],
            "\nA": [12],
            "B": [20, 21],
            " B": [22],
            "\nB": [23],
        }
        return mapping.get(text, [])


def test_align_response_tokens_masks_the_prompt():
    labels, start = align_response_tokens([1, 2, 3], [1, 2, 3, 8, 9])
    assert start == 3
    assert labels == [-100, -100, -100, 8, 9]


def test_align_response_tokens_allows_two_boundary_merges():
    labels, start = align_response_tokens([1, 2, 3, 4], [1, 2, 8, 9])
    assert start == 2
    assert labels == [-100, -100, 8, 9]


def test_choice_token_ids_prefer_single_token_variants():
    assert token_ids_for_choice(ToyTokenizer(), "A") == [10, 11, 12]
    assert token_ids_for_choice(ToyTokenizer(), "B") == [22, 23]


def test_forced_choice_row_uses_option_normalized_log_probability():
    row = forced_choice_row(
        sample_id=7,
        selected="B",
        choice_logprobs={"A": -0.2, "B": -1.2},
        choices=("A", "B"),
        token_count=37,
    )
    expected = -1.2 - np.logaddexp(-0.2, -1.2)
    assert np.isclose(row["mean_answer_logprob"], expected)
    assert row["raw_answer_logprob"] == -1.2
    assert row["token_count"] == 37
    assert row["answer_letter_B"] == 1.0


def test_score_cli_requires_an_explicit_mode():
    args = build_parser().parse_args(
        [
            "--mode",
            "forced-choice",
            "--model",
            "checkpoint",
            "--domain",
            "mmlu",
            "--input",
            "samples.jsonl",
            "--output",
            "sidecar.csv",
        ]
    )
    assert args.mode == "forced-choice"
    assert args.choices == "A,B,C,D"
