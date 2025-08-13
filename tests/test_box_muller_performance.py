import time
from typing import Dict, List

import numpy as np
import pytest

from mc_lab.box_muller import box_muller

# run with pytest -q -m performance -s

# Before performance improvements
# classic: n=50,000 time=0.005073s rate=9856913.1 samples/s
# classic: n=200,000 time=0.015173s rate=13181123.9 samples/s
# .polar:   n=50,000 time=0.004618s rate=10827280.0 samples/s
# polar:   n=200,000 time=0.014559s rate=13737570.5 samples/s
# .compare: classic=0.010038s (19.9 M/s), polar=0.008973s (22.3 M/s)

# After performance improvements
# classic: n=50,000 time=0.001566s rate=31924036.1 samples/s
# classic: n=200,000 time=0.005271s rate=37940952.1 samples/s
# .polar:   n=50,000 time=0.001419s rate=35238689.3 samples/s
# polar:   n=200,000 time=0.006552s rate=30526176.7 samples/s
# .compare: classic=0.004129s (48.4 M/s), polar=0.005474s (36.5 M/s)


@pytest.mark.performance
def test_box_muller_perf_classic():
    """Measure throughput of the classic Box-Muller implementation.

    Informational test that logs timings for a couple of sizes.
    """
    sizes: List[int] = [50_000, 200_000]
    times: Dict[int, float] = {}

    for n in sizes:
        start = time.perf_counter()
        x = box_muller(n, random_state=123, method="classic")
        end = time.perf_counter()
        assert x.shape == (n,)
        times[n] = end - start

    assert all(t > 0 for t in times.values())
    for n in sizes:
        print(
            f"classic: n={n:,} time={times[n]:.6f}s rate={n / max(times[n], 1e-12):.1f} samples/s"
        )


@pytest.mark.performance
def test_box_muller_perf_polar():
    """Measure throughput of the polar (Marsaglia) Box-Muller variant."""
    sizes: List[int] = [50_000, 200_000]
    times: Dict[int, float] = {}

    for n in sizes:
        start = time.perf_counter()
        x = box_muller(n, random_state=123, method="polar")
        end = time.perf_counter()
        assert x.shape == (n,)
        times[n] = end - start

    assert all(t > 0 for t in times.values())
    for n in sizes:
        print(
            f"polar:   n={n:,} time={times[n]:.6f}s rate={n / max(times[n], 1e-12):.1f} samples/s"
        )


@pytest.mark.performance
def test_box_muller_perf_compare_methods():
    """Compare classic vs polar performance on a moderate problem size.

    No strict assertions on which is faster; just report and ensure they both
    complete and produce similar basic stats.
    """
    n = 200_000

    t0 = time.perf_counter()
    x_classic = box_muller(n, random_state=321, method="classic")
    t1 = time.perf_counter()

    x_polar = box_muller(n, random_state=321, method="polar")
    t2 = time.perf_counter()

    dt_classic = t1 - t0
    dt_polar = t2 - t1

    assert x_classic.shape == (n,)
    assert x_polar.shape == (n,)

    for x in (x_classic, x_polar):
        assert np.isfinite(x).all()
        assert abs(x.mean()) < 0.05
        assert 0.9 < x.var() < 1.1

    print(
        "compare: classic={:.6f}s ({:.1f} M/s), polar={:.6f}s ({:.1f} M/s)".format(
            dt_classic,
            (n / max(dt_classic, 1e-12)) / 1e6,
            dt_polar,
            (n / max(dt_polar, 1e-12)) / 1e6,
        )
    )


@pytest.mark.performance
def test_box_muller_perf_polar_njit():
    """Measure throughput of the Numba-accelerated polar variant.

    Skips if Numba is not installed.
    """
    pytest.importorskip("numba")

    sizes: List[int] = [50_000, 200_000]
    times: Dict[int, float] = {}

    for n in sizes:
        start = time.perf_counter()
        x = box_muller(n, random_state=123, method="polar_njit")
        end = time.perf_counter()
        assert x.shape == (n,)
        times[n] = end - start

    assert all(t > 0 for t in times.values())
    for n in sizes:
        print(
            f"polar_njit: n={n:,} time={times[n]:.6f}s rate={n / max(times[n], 1e-12):.1f} samples/s"
        )


@pytest.mark.performance
def test_box_muller_perf_compare_polar_vs_polar_njit():
    """Compare vectorized polar vs Numba-accelerated polar.

    Skips if Numba is not installed. No strict ordering assertion; reports timings.
    """
    pytest.importorskip("numba")

    n = 200_000

    t0 = time.perf_counter()
    x_vec = box_muller(n, random_state=321, method="polar")
    t1 = time.perf_counter()

    x_njit = box_muller(n, random_state=321, method="polar_njit")
    t2 = time.perf_counter()

    dt_vec = t1 - t0
    dt_njit = t2 - t1

    assert x_vec.shape == (n,)
    assert x_njit.shape == (n,)

    for x in (x_vec, x_njit):
        assert np.isfinite(x).all()
        assert abs(x.mean()) < 0.05
        assert 0.9 < x.var() < 1.1

    print(
        "compare (polar): vec={:.6f}s ({:.1f} M/s), njit={:.6f}s ({:.1f} M/s)".format(
            dt_vec,
            (n / max(dt_vec, 1e-12)) / 1e6,
            dt_njit,
            (n / max(dt_njit, 1e-12)) / 1e6,
        )
    )
