import time

import numpy as np
import pytest

from mc_lab.inverse_transform import create_sampler


@pytest.mark.performance
def test_multimodal_chebyshev_vs_numerical_inverse_perf():
    """Compare performance of Chebyshev sampler vs numerical inverse-CDF.

    The PDF is the multimodal density from the paper:
        f(x) = exp(-x^2/2) * (1 + sin^2(3x)) * (1 + cos^2(5x))

    We build:
    - A ChebyshevSampler via the convenience function in scripts/fast_inverse.py
    - A NumericalInverseTransformSampler using an interpolated CDF over [-8, 8]

    This test reports timings and basic sanity checks but does not assert
    strict performance ordering.
    """

    # Load create_multimodal_sampler from scripts/fast_inverse.py dynamically
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "fast_inverse.py"
    spec = importlib.util.spec_from_file_location("_fast_inverse_script", script_path)
    if spec is None or spec.loader is None:
        pytest.skip("Could not load scripts/fast_inverse.py for multimodal sampler")
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    create_multimodal_sampler = _mod.create_multimodal_sampler

    # Define PDF and vectorized CDF for the numerical sampler
    def pdf(x):
        x = np.asarray(x)
        return np.exp(-(x**2) / 2) * (1 + np.sin(3 * x) ** 2) * (1 + np.cos(5 * x) ** 2)

    domain = (-8.0, 8.0)

    # Precompute a high-resolution CDF table for speed and provide a vectorized CDF(x)
    X_GRID = np.linspace(domain[0], domain[1], 20001)
    Y = pdf(X_GRID)
    # Normalize to ensure a proper PDF on the chosen domain
    area = np.trapezoid(Y, X_GRID)
    if area == 0:
        pytest.skip("Multimodal PDF integral over domain is zero; invalid test setup")
    Y /= area
    CDF_GRID = np.cumsum((Y[:-1] + Y[1:]) * np.diff(X_GRID) / 2)
    CDF_GRID = np.concatenate([[0.0], CDF_GRID])
    # Numerical stability: clip to [0,1]
    CDF_GRID = np.clip(CDF_GRID, 0.0, 1.0)

    def cdf(x):
        x = np.asarray(x)
        # Linear interpolation of precomputed CDF
        vals = np.interp(x, X_GRID, CDF_GRID, left=0.0, right=1.0)
        return vals

    # Build samplers
    cheb_sampler = create_multimodal_sampler(domain=list(domain))
    numeric_sampler = create_sampler(
        cdf=cdf,
        x_range=domain,
        method="numerical",
        n_points=2000,  # interpolation resolution inside the numerical sampler
    )

    # Warm-up: ensure any one-time setup is done (Chebyshev precomputation, etc.)
    _ = cheb_sampler.sample(1)
    _ = numeric_sampler.sample(1)

    n = 50_000

    # Time Chebyshev-based sampling
    t0 = time.perf_counter()
    x_cheb = cheb_sampler.sample(n)
    t1 = time.perf_counter()

    # Time numerical inverse-CDF sampling (interpolation)
    x_num = numeric_sampler.sample(n)
    t2 = time.perf_counter()

    dt_cheb = t1 - t0
    dt_num = t2 - t1

    # Sanity checks
    assert x_cheb.shape == (n,)
    assert x_num.shape == (n,)
    assert np.isfinite(x_cheb).all()
    assert np.isfinite(x_num).all()

    # The PDF is symmetric around 0; means should be near 0
    for x in (x_cheb, x_num):
        assert abs(np.mean(x)) < 0.2

    print(
        "multimodal: chebyshev={:.6f}s ({:.1f} K/s), numerical={:.6f}s ({:.1f} K/s), speedup={:.2f}x".format(
            dt_cheb,
            (n / max(dt_cheb, 1e-12)) / 1e3,
            dt_num,
            (n / max(dt_num, 1e-12)) / 1e3,
            (dt_num / max(dt_cheb, 1e-12)),
        )
    )
