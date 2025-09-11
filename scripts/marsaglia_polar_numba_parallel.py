import math

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, parallel=True)
def marsaglia_polar_numba_parallel(n, seed=-1):
    """
    Generate n independent N(0,1) samples using Marsaglia's polar method.
    Parallel version using prange. Writes a pair per iteration for high throughput.

    Parameters
    ----------
    n : int
        Number of samples.
    seed : int
        RNG seed (>=0 for reproducible base state; exact bitwise reproducibility across
        thread counts is not guaranteed).

    Returns
    -------
    out : np.ndarray shape (n,), dtype=float64
        Standard normal samples.
    """
    if seed >= 0:
        np.random.seed(seed)

    out = np.empty(n, dtype=np.float64)
    pairs = (n + 1) // 2  # each iteration generates 2 samples

    # Each iteration generates one accepted pair and writes to disjoint indices
    for i in prange(pairs):
        while True:
            u1 = 2.0 * np.random.random() - 1.0
            u2 = 2.0 * np.random.random() - 1.0
            s = u1 * u1 + u2 * u2
            if (s < 1.0) and (s > 0.0):
                break
        factor = math.sqrt(-2.0 * math.log(s) / s)
        j = 2 * i
        out[j] = u1 * factor
        if j + 1 < n:
            out[j + 1] = u2 * factor

    return out


if __name__ == "__main__":
    # Warmup (compile)
    _ = marsaglia_polar_numba_parallel(10, seed=123)

    # Example large run
    n = 10_000_000
    x = marsaglia_polar_numba_parallel(n, seed=123)
    print("mean:", x.mean(), "var:", x.var())
