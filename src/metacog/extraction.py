"""Final-answer-token transformer-block activation extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .io import sha256_file, stable_hash, write_json, write_jsonl
from .prompts import answer_prompt
from .thresholding import quadrant_labels


CHAT_ENDINGS = (
    "<|im_end|>\n",
    "<|im_end|>",
    "<|eot_id|>",
    "<|endoftext|>",
)


def strip_terminal_markers(text: str, *, strip_period: bool = True) -> str:
    for marker in CHAT_ENDINGS:
        if text.endswith(marker):
            text = text[: -len(marker)]
            break
    if strip_period and text.endswith("."):
        text = text[:-1]
    return text


def render_answer_sequence(tokenizer, record: dict, domain: str) -> str:
    messages = [
        {"role": "user", "content": answer_prompt(record, domain)},
        {"role": "assistant", "content": str(record.get("response", record.get("answer", "")))},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    return strip_terminal_markers(rendered)


def parse_layers(specification: str, number_of_layers: int) -> list[int]:
    if specification.strip().lower() == "all":
        return list(range(number_of_layers))
    layers = [int(value.strip()) for value in specification.split(",") if value.strip()]
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("layers must be a non-empty set of unique indices")
    if min(layers) < 0 or max(layers) >= number_of_layers:
        raise ValueError(f"layer index outside [0, {number_of_layers - 1}]")
    return layers


def transformer_blocks(model) -> Sequence:
    candidates = (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers"))
    for parent_name, child_name in candidates:
        parent = getattr(model, parent_name, None)
        if parent is not None and hasattr(parent, child_name):
            return getattr(parent, child_name)
    raise AttributeError("unable to locate transformer block modules")


def extract_final_token_states(
    model,
    tokenizer,
    token_sequences: Sequence[Sequence[int]],
    *,
    layers: Sequence[int],
    batch_size: int,
    device: str,
    storage_dtype: str = "float16",
) -> np.ndarray:
    import torch
    from tqdm import tqdm

    dtype = {"float16": torch.float16, "float32": torch.float32}[storage_dtype]
    blocks = transformer_blocks(model)
    captured: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Left padding makes the last tensor position the final answer token.
            value = hidden[:, -1, :].detach().to(dtype=dtype).cpu().numpy()
            captured[layer].append(value)

        return hook

    for layer in layers:
        handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
    try:
        for begin in tqdm(range(0, len(token_sequences), batch_size), desc="activation batches"):
            batch = token_sequences[begin : begin + batch_size]
            features = [{"input_ids": list(ids)} for ids in batch]
            inputs = tokenizer.pad(features, padding=True, return_tensors="pt")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.no_grad():
                model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    matrices = []
    for layer in layers:
        matrix = np.concatenate(captured[layer], axis=0)
        if len(matrix) != len(token_sequences):
            raise RuntimeError(f"layer {layer} produced an incomplete activation matrix")
        matrices.append(matrix)
    return np.stack(matrices, axis=1)


def read_processed_records(path: Path, max_samples: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records if max_samples is None else records[:max_samples]


def write_activation_artifact(
    output_dir: Path,
    *,
    model_key: str,
    public_model_id: str,
    model_revision: str | None,
    domain: str,
    source_name: str,
    layers: Sequence[int],
    activations: np.ndarray,
    records: Sequence[dict],
    token_counts: Sequence[int],
    device_dtype: str,
    storage_dtype: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    oc = np.asarray([int(row["objective_correctness"]) for row in records], dtype=np.int8)
    sj = np.asarray([int(row["self_judgement"]) for row in records], dtype=np.int8)
    p_sj = np.asarray([float(row["p_self_judgement"]) for row in records], dtype=np.float32)
    sample_ids = np.arange(len(records), dtype=np.int32)
    quadrants = quadrant_labels(oc, sj)

    samples = []
    for sample_id, (record, token_count, quadrant) in enumerate(
        zip(records, token_counts, quadrants)
    ):
        samples.append(
            {
                "sample_id": sample_id,
                "model": model_key,
                "domain": domain,
                "question_id": str(record["question_id"]),
                "pair_id": str(record.get("pair_id", record["question_id"])),
                "question": record["question"],
                "response": record["response"],
                "answer": record.get("answer", record["response"]),
                "objective_correctness": int(record["objective_correctness"]),
                "p_self_judgement": float(record["p_self_judgement"]),
                "self_judgement": int(record["self_judgement"]),
                "quadrant": str(quadrant),
                "token_count": int(token_count),
            }
        )

    metadata = {
        "schema_version": "activation_v1",
        "model": model_key,
        "public_model_id": public_model_id,
        "model_revision": model_revision,
        "domain": domain,
        "source_name": source_name,
        "activation_site": "final_non_padding_answer_token_transformer_block_output",
        "device_dtype": device_dtype,
        "storage_dtype": storage_dtype,
        "num_samples": len(records),
        "layers": [int(layer) for layer in layers],
        "hidden_size": int(activations.shape[-1]),
    }
    metadata_path = output_dir / "metadata.json"
    samples_path = output_dir / "samples.jsonl"
    arrays_path = output_dir / f"layers_{layers[0]}_{layers[-1]}.npz"
    write_json(metadata_path, metadata)
    write_jsonl(samples_path, samples)
    np.savez(
        arrays_path,
        activations=activations,
        layers=np.asarray(layers, dtype=np.int16),
        sample_id=sample_ids,
        objective_correctness=oc,
        self_judgement=sj,
        p_self_judgement=p_sj,
    )
    write_json(
        output_dir / "checksums.json",
        {path.name: sha256_file(path) for path in (metadata_path, samples_path, arrays_path)},
    )
