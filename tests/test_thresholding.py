import numpy as np

from metacog.thresholding import (
    apply_symmetric_confidence,
    paired_confidence_mask,
    quadrant_labels,
)


def test_symmetric_confidence_removes_only_open_uncertainty_band():
    result = apply_symmetric_confidence(np.asarray([0.0, 0.3, 0.31, 0.69, 0.7, 1.0]), 0.7)
    assert result.keep.tolist() == [True, True, False, False, True, True]
    assert result.labels.tolist() == [0, 0, 0, 0, 1, 1]


def test_pair_filter_drops_both_rows_when_one_is_uncertain():
    probabilities = np.asarray([0.1, 0.8, 0.2, 0.6])
    pairs = np.asarray(["a", "a", "b", "b"])
    assert paired_confidence_mask(probabilities, pairs, 0.7).tolist() == [True, True, False, False]


def test_quadrant_labels_follow_oc_by_sj_design():
    labels = quadrant_labels(np.asarray([1, 1, 0, 0]), np.asarray([1, 0, 1, 0]))
    assert labels.tolist() == ["A", "B", "C", "D"]
