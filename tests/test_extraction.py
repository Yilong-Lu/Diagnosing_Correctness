from metacog.extraction import parse_layers, strip_terminal_markers
from metacog.prompts import answer_prompt


def test_terminal_chat_marker_and_period_are_removed():
    assert strip_terminal_markers("answer.<|eot_id|>") == "answer"
    assert strip_terminal_markers("answer.<|im_end|>\n") == "answer"
    assert strip_terminal_markers("answer.") == "answer"


def test_layer_parser():
    assert parse_layers("all", 3) == [0, 1, 2]
    assert parse_layers("0,2", 3) == [0, 2]


def test_ood_answer_prompt_requests_a_single_letter():
    prompt = answer_prompt({"question": "Q", "prompt": "Q\nA. x\nB. y"}, "mmlu")
    assert prompt.endswith("Answer only with a single letter.")
    assert prompt.count("Answer only with a single letter.") == 1
