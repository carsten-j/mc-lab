import numpy as np

from mc_lab.transformation_methods import (
    sample_cauchy_via_tangent,
    sample_exponential_via_inverse,
)


def test_exponential_mean_matches_rate():
    n = 300_000
    rate = 2.5
    x = sample_exponential_via_inverse(n, rate=rate, random_state=123)
    assert x.shape == (n,)
    # E[X] = 1/rate
    assert np.isclose(x.mean(), 1.0 / rate, atol=0.02)


def test_exponential_reproducibility():
    n = 10_000
    a = sample_exponential_via_inverse(n, rate=1.7, random_state=42)
    b = sample_exponential_via_inverse(n, rate=1.7, random_state=42)
    assert np.allclose(a, b)


def test_cauchy_median_and_half_mass_within_scale():
    n = 300_000
    loc = -1.2
    scale = 0.8
    x = sample_cauchy_via_tangent(n, location=loc, scale=scale, random_state=7)
    assert x.shape == (n,)

    # Median should be close to location
    med = np.median(x)
    assert np.isclose(med, loc, atol=0.03)

    # For Cauchy(loc, scale): P(|X - loc| <= scale) = 0.5
    frac = np.mean(np.abs(x - loc) <= scale)
    assert np.isclose(frac, 0.5, atol=0.02)


def test_cauchy_reproducibility():
    n = 20_000
    a = sample_cauchy_via_tangent(n, location=0.0, scale=2.0, random_state=123)
    b = sample_cauchy_via_tangent(n, location=0.0, scale=2.0, random_state=123)
    assert np.allclose(a, b)
