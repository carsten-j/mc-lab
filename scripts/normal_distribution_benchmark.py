"""
Comprehensive performance comparison of different normal distribution generators.
This script runs all available implementations and provides a side-by-side comparison.
"""

import sys
import timeit

# Add the scripts directory to path so we can import our implementations
sys.path.append("/home/carstenj/dev/mc-lab/scripts")

import numpy as np
from marsaglia_polar import marsaglia_polar
from marsaglia_polar_numba import marsaglia_polar_numba
from marsaglia_polar_numba_parallel import marsaglia_polar_numba_parallel
from numpy_normal import numpy_normal_randn, numpy_normal_standard


def run_timing_comparison(n=10_000_000, runs=5):
    """
    Run comprehensive timing comparison of all normal distribution generators.

    Parameters
    ----------
    n : int
        Number of samples to generate per test.
    runs : int
        Number of timing runs to average over.
    """
    print("=" * 80)
    print("COMPREHENSIVE NORMAL DISTRIBUTION PERFORMANCE COMPARISON")
    print(f"Testing with n={n:,} samples, {runs} runs each")
    print("=" * 80)

    results = {}

    # Test each implementation
    implementations = [
        (
            "NumPy Pure (marsaglia_polar)",
            lambda: marsaglia_polar(n, rng=np.random.default_rng(123)),
        ),
        ("NumPy Generator.standard_normal", lambda: numpy_normal_standard(n, seed=123)),
        ("NumPy Legacy randn", lambda: numpy_normal_randn(n, seed=123)),
        ("Numba Single-threaded", lambda: marsaglia_polar_numba(n, seed=123)),
        ("Numba Parallel", lambda: marsaglia_polar_numba_parallel(n, seed=123)),
    ]

    # Warmup (especially important for Numba)
    print("Warming up implementations...")
    for name, func in implementations:
        if "Numba" in name:
            _ = func.__func__(100, seed=123) if hasattr(func, "__func__") else func()

    print("\nRunning benchmarks...\n")

    for name, func in implementations:
        try:
            # Time the function
            total_time = timeit.timeit(func, number=runs)
            average_time = total_time / runs
            samples_per_sec = n / average_time

            # Verify correctness with a single run
            x = func()
            mean_val = x.mean()
            var_val = x.var()

            results[name] = {
                "avg_time": average_time,
                "samples_per_sec": samples_per_sec,
                "mean": mean_val,
                "var": var_val,
            }

            print(f"{name}:")
            print(f"  Average time per run: {average_time:.4f} seconds")
            print(f"  Samples per second:   {samples_per_sec:>12,.0f}")
            print(f"  Verification - mean: {mean_val:8.6f}, var: {var_val:.6f}")
            print()

        except Exception as e:
            print(f"{name}: ERROR - {e}")
            print()

    # Summary comparison
    print("=" * 80)
    print("PERFORMANCE RANKING (fastest to slowest)")
    print("=" * 80)

    # Sort by samples per second (descending)
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["samples_per_sec"], reverse=True
    )

    if sorted_results:
        fastest_speed = sorted_results[0][1]["samples_per_sec"]

        for i, (name, data) in enumerate(sorted_results, 1):
            speedup = data["samples_per_sec"] / sorted_results[-1][1]["samples_per_sec"]
            relative_to_fastest = fastest_speed / data["samples_per_sec"]

            print(f"{i}. {name}")
            print(f"   {data['samples_per_sec']:>12,.0f} samples/sec")
            print(f"   {speedup:>12.2f}x faster than slowest")
            if i > 1:
                print(f"   {relative_to_fastest:>12.2f}x slower than fastest")
            else:
                print(f"   {'FASTEST':>12}")
            print()

    return results


if __name__ == "__main__":
    results = run_timing_comparison()
