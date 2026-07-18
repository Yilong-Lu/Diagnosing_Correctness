import math

import pytest

from metacog.judgement import (
    JUDGEMENT_PROMPT,
    build_judgement_messages,
    classify_topk_coverage,
    full_binary_probability,
    historical_top4_probability,
    resolve_literal_token_ids,
)


class ToyTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"Yes": [10], "No": [20], "Maybe": [30, 31]}[text]


def test_judgement_messages_preserve_answer_context_and_exact_question():
    messages = build_judgement_messages(
        {"question": "Who?", "response": "Actor", "objective_correctness": 1},
        "movies",
    )
    assert messages[0]["content"] == "Who?\nReturn only the name of the actor."
    assert messages[1]["content"] == "Actor"
    assert messages[2]["content"] == JUDGEMENT_PROMPT


def test_literal_token_policy_and_topk_coverage():
    assert resolve_literal_token_ids(ToyTokenizer()) == {"Yes": 10, "No": 20}
    assert classify_topk_coverage([10, 20, 30], yes_token_id=10, no_token_id=20) == "both"
    assert classify_topk_coverage([10, 30], yes_token_id=10, no_token_id=20) == "yes_only"
    assert classify_topk_coverage([20, 30], yes_token_id=10, no_token_id=20) == "no_only"
    assert classify_topk_coverage([30], yes_token_id=10, no_token_id=20) == "neither"
    with pytest.raises(ValueError, match="must encode as one token"):
        resolve_literal_token_ids(type("Bad", (), {"encode": lambda *args, **kwargs: [1, 2]})())


def test_full_and_historical_probabilities_match_their_definitions():
    assert full_binary_probability(2.0, 1.0) == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert historical_top4_probability(-0.2, -2.0, "yes_only") == pytest.approx(
        math.exp(-0.2)
    )
    assert historical_top4_probability(-2.0, -0.2, "no_only") == pytest.approx(
        1.0 - math.exp(-0.2)
    )
    assert historical_top4_probability(-1.0, -1.0, "neither") == 0.5
