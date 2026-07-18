import numpy as np

from metacog.directions import fit_factorial_directions, project


def test_factorial_contrasts_recover_additive_components():
    oc = np.asarray([1, 1, 0, 0], dtype=np.int8)
    sj = np.asarray([1, 0, 1, 0], dtype=np.int8)
    x = np.column_stack([2 * oc - 1, 2 * sj - 1]).astype(float)
    fitted = fit_factorial_directions(x, oc, sj)
    np.testing.assert_allclose(fitted.truth, [2.0, 0.0])
    np.testing.assert_allclose(fitted.meta, [0.0, 2.0])
    np.testing.assert_allclose(fitted.mixed, [2.0, 2.0])
    np.testing.assert_allclose(fitted.oc_only, [2.0, 0.0])


def test_projection_uses_source_center():
    scores = project(np.asarray([[2.0, 3.0]]), np.asarray([1.0, 0.0]), np.asarray([1.0, 1.0]))
    np.testing.assert_allclose(scores, [1.0])
