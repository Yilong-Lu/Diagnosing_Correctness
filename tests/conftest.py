from __future__ import annotations

import numpy as np
import pytest

from metacog.schema import ActivationBundle


@pytest.fixture
def synthetic_bundle_factory(tmp_path):
    def build(model: str = "toy", domain: str = "math", shift: float = 0.0):
        rows = []
        activations = []
        oc_values = []
        sj_values = []
        question_ids = []
        sample_ids = []
        sample_id = 0
        # Every question contributes one correct and one incorrect response.
        # Across questions, both SJ states occur within each OC class.
        pair_patterns = [
            ((1, 1), (0, 1)),
            ((1, 0), (0, 0)),
            ((1, 1), (0, 0)),
            ((1, 0), (0, 1)),
        ] * 3
        for question, pair in enumerate(pair_patterns):
            question_id = f"q{question:02d}"
            for oc, sj in pair:
                base = np.asarray(
                    [2 * oc - 1, 2 * sj - 1, 0.25 * (2 * oc - 1) * (2 * sj - 1)],
                    dtype=np.float32,
                )
                layer_values = np.stack([base + shift + layer * 0.01 for layer in range(3)])
                activations.append(layer_values)
                oc_values.append(oc)
                sj_values.append(sj)
                question_ids.append(question_id)
                sample_ids.append(sample_id)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "question_id": question_id,
                        "question": f"Question {question_id}",
                    }
                )
                sample_id += 1
        oc_array = np.asarray(oc_values, dtype=np.int8)
        sj_array = np.asarray(sj_values, dtype=np.int8)
        return ActivationBundle(
            model=model,
            domain=domain,
            activations=np.asarray(activations, dtype=np.float32),
            layers=np.asarray([0, 1, 2], dtype=np.int16),
            sample_ids=np.asarray(sample_ids, dtype=np.int32),
            question_ids=np.asarray(question_ids),
            objective_correctness=oc_array,
            self_judgement=sj_array,
            p_self_judgement=np.where(sj_array == 1, 0.9, 0.1),
            records=rows,
            source_dir=tmp_path,
        )

    return build
