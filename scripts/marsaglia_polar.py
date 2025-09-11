import timeit

import numpy as np


def marsaglia_polar(n, rng=None, dtype=np.float64, max_batch=4_000_000):
    """
    Generate n independent standard normal samples using Marsaglia's polar method.

    Parameters
    ----------
    n : int
        Number of N(0,1) samples to generate.
    rng : numpy.random.Generator or None
        RNG to use. If None, uses np.random.default_rng().
    dtype : data-type
        Floating dtype for outputs. Typically np.float64 (default) or np.float32.
    max_batch : int
        Max number of proposals generated per batch (per U1 or U2).
        Controls peak memory usage and performance.

    Returns
    -------
    out : np.ndarray, shape (n,), dtype=dtype
        Independent N(0,1) samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    out = np.empty(n, dtype=dtype)
    filled = 0
    # Acceptance probability is pi/4 ≈ 0.785; conservatively over-propose by ~1/0.75
    ACCEPT_EST = 0.75

    while filled < n:
        need = n - filled
        # Propose enough to likely fill what's needed, bounded by max_batch
        m = min(max_batch, max(int(need / ACCEPT_EST) + 16, 1024))

        # Propose uniformly in [-1, 1]^2
        u1 = rng.uniform(-1.0, 1.0, size=m).astype(dtype, copy=False)
        u2 = rng.uniform(-1.0, 1.0, size=m).astype(dtype, copy=False)
        s = u1 * u1 + u2 * u2

        # Accept points strictly inside the unit circle and exclude s=0 to avoid division by zero
        mask = (s > 0.0) & (s < 1.0)
        if not mask.any():
            continue

        u1a = u1[mask]
        u2a = u2[mask]
        sa = s[mask]

        # Transform to normals: factor = sqrt(-2 ln s / s)
        # Note: sa in (0,1), safe for log
        factor = np.sqrt(-2.0 * np.log(sa) / sa, dtype=dtype)
        x1 = u1a * factor
        x2 = u2a * factor

        # Fill the output array efficiently
        k = x1.size
        # How many full pairs can we use?
        pair_count = min(k, (need // 2))
        if pair_count > 0:
            # Interleave x1 and x2 into out[filled:filled+2*pair_count]
            out[filled : filled + 2 * pair_count : 2] = x1[:pair_count]
            out[filled + 1 : filled + 2 * pair_count : 2] = x2[:pair_count]
            filled += 2 * pair_count

        # If we still need one more sample and we still have leftovers, take x1 of the next pair
        if filled < n and k > pair_count:
            out[filled] = x1[pair_count]
            filled += 1

    return out


# Example usage
if __name__ == "__main__":
    rng = np.random.default_rng(12345)
    n = 10_000_000

    # Time the marsaglia_polar function
    def time_marsaglia():
        return marsaglia_polar(n, rng=rng, dtype=np.float64, max_batch=4_000_000)

    # Use timeit to measure execution time
    number_of_runs = 5
    total_time = timeit.timeit(time_marsaglia, number=number_of_runs)
    average_time = total_time / number_of_runs

    print(f"Timing results for marsaglia_polar with n={n:,} samples:")
    print(f"Number of runs: {number_of_runs}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per run: {average_time:.4f} seconds")
    print(f"Samples per second: {n / average_time:,.0f}")

    # Generate one sample to verify correctness
    x = marsaglia_polar(n, rng=rng, dtype=np.float64, max_batch=4_000_000)
    print(f"Verification - mean: {x.mean():.6f}, var: {x.var():.6f}")
