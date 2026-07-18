"""Recompute literal Yes/No logits for frozen self-judgement records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..io import read_jsonl, write_jsonl
from ..judgement import score_judgement_logits


def _read_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("expected a JSON list or JSON Lines input")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Public model ID or local checkpoint path")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--domain", choices=["math", "movies"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


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
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    ).eval().to(args.device)
    rows = _read_rows(args.input)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    scored = score_judgement_logits(
        model,
        tokenizer,
        rows,
        domain=args.domain,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    write_jsonl(args.output, scored)


if __name__ == "__main__":
    main()
