"""Extract final-answer-token hidden states from processed response records."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..extraction import (
    extract_final_token_states,
    parse_layers,
    read_processed_records,
    render_answer_sequence,
    transformer_blocks,
    write_activation_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model", required=True, help="Public model ID or local checkpoint path")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
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

    records = read_processed_records(args.input, args.max_samples)
    rendered = [render_answer_sequence(tokenizer, record, args.domain) for record in records]
    token_sequences = [tokenizer.encode(text, add_special_tokens=False) for text in rendered]
    too_long = [index for index, ids in enumerate(token_sequences) if len(ids) > args.max_length]
    if too_long:
        raise ValueError(f"{len(too_long)} sequences exceed --max-length; first index {too_long[0]}")
    layers = parse_layers(args.layers, len(transformer_blocks(model)))
    activations = extract_final_token_states(
        model,
        tokenizer,
        token_sequences,
        layers=layers,
        batch_size=args.batch_size,
        device=args.device,
        storage_dtype=args.storage_dtype,
    )
    write_activation_artifact(
        args.output / args.model_key / args.domain,
        model_key=args.model_key,
        public_model_id=args.model if not Path(args.model).exists() else Path(args.model).name,
        model_revision=args.revision,
        domain=args.domain,
        source_name=args.input.name,
        layers=layers,
        activations=activations,
        records=records,
        token_counts=[len(ids) for ids in token_sequences],
        device_dtype=args.device_dtype,
        storage_dtype=args.storage_dtype,
    )


if __name__ == "__main__":
    main()
