"""Score free-response answers or forced-choice options for nuisance controls."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from ..extraction import render_answer_sequence
from ..io import read_jsonl, write_json
from ..prompts import answer_prompt
from ..scoring import align_response_tokens, forced_choice_row, token_ids_for_choice


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["free-response", "forced-choice"])
    parser.add_argument("--model", required=True, help="Public model ID or local checkpoint path")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--domain", required=True, choices=["math", "movies", "mmlu", "truthfulqa_binary"])
    parser.add_argument("--input", type=Path, required=True, help="Processed JSON Lines records")
    parser.add_argument("--output", type=Path, required=True, help="Output nuisance CSV")
    parser.add_argument("--choices", default="A,B,C,D")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("answer scoring produced no rows")
    columns = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _free_response_rows(model, tokenizer, records: list[dict], args) -> list[dict]:
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from tqdm import tqdm

    tokenizer.padding_side = "right"
    encoded: list[list[int]] = []
    labels: list[list[int]] = []
    for record in records:
        user = [{"role": "user", "content": answer_prompt(record, args.domain)}]
        prefix = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        full = render_answer_sequence(tokenizer, record, args.domain)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = tokenizer.encode(full, add_special_tokens=False)
        if len(full_ids) > args.max_length:
            raise ValueError(f"sample {record['sample_id']} exceeds --max-length={args.max_length}")
        row_labels, _start = align_response_tokens(prefix_ids, full_ids)
        encoded.append(full_ids)
        labels.append(row_labels)

    rows: list[dict] = []
    loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    for start in tqdm(range(0, len(records), args.batch_size), desc="answer likelihood"):
        batch_ids = [torch.tensor(value, dtype=torch.long) for value in encoded[start : start + args.batch_size]]
        batch_labels = [torch.tensor(value, dtype=torch.long) for value in labels[start : start + args.batch_size]]
        input_ids = pad_sequence(batch_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(args.device)
        label_ids = pad_sequence(batch_labels, batch_first=True, padding_value=-100).to(args.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = label_ids[:, 1:].contiguous()
        losses = loss(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        losses = losses.view(shifted_labels.shape)
        for offset in range(len(batch_ids)):
            valid = shifted_labels[offset] != -100
            values = losses[offset][valid]
            count = int(values.numel())
            if count == 0:
                raise ValueError(f"sample {records[start + offset]['sample_id']} has no scored answer tokens")
            total = -float(values.sum().item())
            record = records[start + offset]
            rows.append(
                {
                    "sample_id": int(record["sample_id"]),
                    "question_id": str(record.get("question_id", "")),
                    "mean_answer_logprob": total / count,
                    "sum_answer_logprob": total,
                    "token_count": count,
                }
            )
    return rows


def _forced_choice_rows(model, tokenizer, records: list[dict], args) -> list[dict]:
    import torch
    import torch.nn.functional as functional
    from tqdm import tqdm

    choices = tuple(value.strip() for value in args.choices.split(",") if value.strip())
    if not choices:
        raise ValueError("--choices must contain at least one option")
    choice_ids = {choice: token_ids_for_choice(tokenizer, choice) for choice in choices}
    if any(not ids for ids in choice_ids.values()):
        raise ValueError("one or more forced choices could not be tokenized")
    tokenizer.padding_side = "left"
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": answer_prompt(record, args.domain)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for record in records
    ]
    lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    if lengths and max(lengths) > args.max_length:
        raise ValueError(f"a forced-choice prompt exceeds --max-length={args.max_length}")
    sequence_lengths = [
        len(
            tokenizer.encode(
                render_answer_sequence(tokenizer, record, args.domain),
                add_special_tokens=False,
            )
        )
        for record in records
    ]

    rows: list[dict] = []
    for start in tqdm(range(0, len(records), args.batch_size), desc="option likelihood"):
        batch = prompts[start : start + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(args.device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :].float()
        logprobs = functional.log_softmax(logits, dim=-1)
        for offset, record in enumerate(records[start : start + args.batch_size]):
            scores = {
                choice: float(torch.logsumexp(logprobs[offset, torch.tensor(ids, device=args.device)], dim=0).cpu())
                for choice, ids in choice_ids.items()
            }
            selected = str(record.get("option_label", record.get("answer", record.get("response", "")))).strip().upper()
            row = forced_choice_row(
                sample_id=int(record["sample_id"]),
                selected=selected,
                choice_logprobs=scores,
                choices=choices,
                token_count=sequence_lengths[start + offset],
            )
            row["question_id"] = str(record.get("question_id", ""))
            rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.device_dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    ).eval().to(args.device)
    records = read_jsonl(args.input)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if args.mode == "free-response":
        if args.domain not in {"math", "movies"}:
            raise ValueError("free-response scoring supports Math and Movies")
        rows = _free_response_rows(model, tokenizer, records, args)
    else:
        if args.domain not in {"mmlu", "truthfulqa_binary"}:
            raise ValueError("forced-choice scoring supports MMLU and binary TruthfulQA")
        rows = _forced_choice_rows(model, tokenizer, records, args)
    _write_csv(args.output, rows)
    write_json(
        args.output.with_suffix(".metadata.json"),
        {
            "mode": args.mode,
            "domain": args.domain,
            "model": args.model if not Path(args.model).exists() else Path(args.model).name,
            "revision": args.revision,
            "rows": len(rows),
            "input": args.input.name,
            "output": args.output.name,
            "device_dtype": args.device_dtype,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
    )


if __name__ == "__main__":
    main()
