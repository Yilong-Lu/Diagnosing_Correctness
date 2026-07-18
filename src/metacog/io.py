"""Artifact readers and writers with path-independent metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .schema import ActivationBundle


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _array(npz: Mapping[str, np.ndarray], preferred: str, legacy: str) -> np.ndarray:
    if preferred in npz:
        return np.asarray(npz[preferred])
    if legacy in npz:
        return np.asarray(npz[legacy])
    raise KeyError(f"activation artifact lacks {preferred!r}")


def load_activation_bundle(directory: Path) -> ActivationBundle:
    """Load the clean artifact schema and the compatible historical schema."""

    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    records = read_jsonl(directory / "samples.jsonl")
    npz_paths = sorted(directory.glob("layers_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"no layers_*.npz file found in {directory}")

    activation_parts = []
    layer_parts = []
    first_arrays: dict[str, np.ndarray] | None = None
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        activation_parts.append(arrays["activations"])
        layer_parts.append(arrays["layers"])
        if first_arrays is None:
            first_arrays = arrays
        elif not np.array_equal(arrays["sample_id"], first_arrays["sample_id"]):
            raise ValueError("activation layer shards have inconsistent sample order")

    assert first_arrays is not None
    activations = np.concatenate(activation_parts, axis=1)
    layers = np.concatenate(layer_parts)
    order = np.argsort(layers)
    activations = activations[:, order, :]
    layers = layers[order]

    sample_ids = np.asarray(first_arrays["sample_id"])
    record_by_id = {int(row["sample_id"]): row for row in records}
    ordered_records = [record_by_id[int(sample_id)] for sample_id in sample_ids]
    question_ids = np.asarray(
        [row.get("question_id", stable_hash(str(row["question"]))) for row in ordered_records],
        dtype=str,
    )

    return ActivationBundle(
        model=str(metadata.get("model", metadata.get("model_name", "unknown"))),
        domain=str(metadata["domain"]),
        activations=activations,
        layers=layers,
        sample_ids=sample_ids,
        question_ids=question_ids,
        objective_correctness=_array(first_arrays, "objective_correctness", "GT").astype(np.int8),
        self_judgement=_array(first_arrays, "self_judgement", "MJ").astype(np.int8),
        p_self_judgement=_array(
            first_arrays, "p_self_judgement", "p_judgement"
        ).astype(np.float64),
        records=ordered_records,
        source_dir=directory,
    )
