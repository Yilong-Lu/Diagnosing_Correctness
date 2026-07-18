"""Prompt rendering used for activation and likelihood extraction."""

from __future__ import annotations

MATH_ANSWER_PROMPT = """Question: {question}
Strictly output your response in the following exact format:
Reasoning: A concise reasoning focusing on the core steps needed to solve the problem.
Answer: <final numeric answer only>."""

MOVIES_ANSWER_PROMPT = "{question}\nReturn only the name of the actor."
CHOICE_ANSWER_SUFFIX = "Answer only with a single letter."

def answer_prompt(record: dict, domain: str) -> str:
    if domain == "math":
        return MATH_ANSWER_PROMPT.format(question=record["question"])
    if domain == "movies":
        return MOVIES_ANSWER_PROMPT.format(question=record["question"])
    if domain in {"mmlu", "truthfulqa_binary"}:
        prompt = str(record.get("prompt", record["question"])).rstrip()
        return prompt if prompt.endswith(CHOICE_ANSWER_SUFFIX) else f"{prompt}\n{CHOICE_ANSWER_SUFFIX}"
    raise ValueError(f"unsupported domain: {domain}")
