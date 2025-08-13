import numpy as np

from mc_lab.box_muller import box_muller


def test_box_muller_classic_stats():
    n = 200_000
    x = box_muller(n, random_state=42, method="classic")
    assert x.shape == (n,)
    # Check mean ~ 0 and var ~ 1 (loose tolerances for randomness)
    assert np.isclose(x.mean(), 0.0, atol=0.02)
    assert np.isclose(x.var(), 1.0, atol=0.03)


def test_box_muller_polar_stats():
    n = 200_000
    x = box_muller(n, random_state=123, method="polar")
    assert x.shape == (n,)
    assert np.isclose(x.mean(), 0.0, atol=0.02)
    assert np.isclose(x.var(), 1.0, atol=0.03)


def test_box_muller_return_pairs_classic_shape_and_values():
    m = 10_001
    pairs = box_muller(2 * m, random_state=7, method="classic", return_pairs=True)
    assert pairs.shape == (m, 2)
    # flatten should produce same numbers as direct call
    flat = box_muller(2 * m, random_state=7, method="classic", return_pairs=False)
    assert np.allclose(pairs.ravel()[: 2 * m], flat)


def test_box_muller_return_pairs_polar_shape():
    m = 5_000
    pairs = box_muller(2 * m, random_state=1234, method="polar", return_pairs=True)
    assert pairs.shape == (m, 2)
