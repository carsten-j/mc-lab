import math
import timeit

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def marsaglia_polar_numba(n, seed=-1):
    """
    Generate n independent N(0,1) samples using Marsaglia's polar method.
    Single-threaded version, JIT-compiled with numba.

    Parameters
    ----------
    n : int
        Number of samples.
    seed : int
        RNG seed (>=0 for reproducible results; <0 to leave seed unchanged).

    Returns
    -------
    out : np.ndarray shape (n,), dtype=float64
        Standard normal samples.
    """
    if seed >= 0:
        np.random.seed(seed)

    out = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        # Rejection until inside unit circle
        while True:
            u1 = 2.0 * np.random.random() - 1.0
            u2 = 2.0 * np.random.random() - 1.0
            s = u1 * u1 + u2 * u2
            if (s < 1.0) and (s > 0.0):
                break

        # Transform
        factor = math.sqrt(-2.0 * math.log(s) / s)
        out[i] = u1 * factor
        i += 1
        if i < n:
            out[i] = u2 * factor
            i += 1

    return out


if __name__ == "__main__":
    # Warmup (compile)
    _ = marsaglia_polar_numba(10, seed=123)

    # Example large run with timing
    n = 10_000_000

    # Time the marsaglia_polar_numba function
    def time_marsaglia_numba():
        return marsaglia_polar_numba(n, seed=123)

    # Use timeit to measure execution time
    number_of_runs = 5
    total_time = timeit.timeit(time_marsaglia_numba, number=number_of_runs)
    average_time = total_time / number_of_runs

    print(f"Timing results for marsaglia_polar_numba with n={n:,} samples:")
    print(f"Number of runs: {number_of_runs}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per run: {average_time:.4f} seconds")
    print(f"Samples per second: {n / average_time:,.0f}")

    # Generate one sample to verify correctness
    x = marsaglia_polar_numba(n, seed=123)
    print(f"Verification - mean: {x.mean():.6f}, var: {x.var():.6f}")
