import timeit

import numpy as np


def numpy_normal_standard(n, seed=-1):
    """
    Generate n independent N(0,1) samples using NumPy's built-in normal generator.
    NumPy internally uses optimized algorithms like Ziggurat for high performance.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    seed : int
        RNG seed (>=0 for reproducible results; <0 to leave seed unchanged).

    Returns
    -------
    out : np.ndarray, shape (n,), dtype=float64
        Standard normal samples.
    """
    if seed >= 0:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    return rng.standard_normal(n, dtype=np.float64)


def numpy_normal_randn(n, seed=-1):
    """
    Generate n independent N(0,1) samples using NumPy's randn function.
    Uses the legacy random interface but still efficient.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    seed : int
        RNG seed (>=0 for reproducible results; <0 to leave seed unchanged).

    Returns
    -------
    out : np.ndarray, shape (n,), dtype=float64
        Standard normal samples.
    """
    if seed >= 0:
        np.random.seed(seed)

    return np.random.randn(n)


if __name__ == "__main__":
    n = 10_000_000

    print("=" * 60)
    print("NUMPY NORMAL DISTRIBUTION PERFORMANCE COMPARISON")
    print("=" * 60)

    # Time the new Generator interface (standard_normal)
    def time_numpy_standard():
        return numpy_normal_standard(n, seed=123)

    number_of_runs = 5
    total_time = timeit.timeit(time_numpy_standard, number=number_of_runs)
    average_time = total_time / number_of_runs

    print("\nNumPy Generator.standard_normal():")
    print(f"Timing results for numpy_normal_standard with n={n:,} samples:")
    print(f"Number of runs: {number_of_runs}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per run: {average_time:.4f} seconds")
    print(f"Samples per second: {n / average_time:,.0f}")

    # Verify correctness
    x = numpy_normal_standard(n, seed=123)
    print(f"Verification - mean: {x.mean():.6f}, var: {x.var():.6f}")

    # Time the legacy randn interface
    def time_numpy_randn():
        return numpy_normal_randn(n, seed=123)

    total_time_randn = timeit.timeit(time_numpy_randn, number=number_of_runs)
    average_time_randn = total_time_randn / number_of_runs

    print("\nNumPy Legacy np.random.randn():")
    print(f"Timing results for numpy_normal_randn with n={n:,} samples:")
    print(f"Number of runs: {number_of_runs}")
    print(f"Total time: {total_time_randn:.4f} seconds")
    print(f"Average time per run: {average_time_randn:.4f} seconds")
    print(f"Samples per second: {n / average_time_randn:,.0f}")

    # Verify correctness
    x_randn = numpy_normal_randn(n, seed=123)
    print(f"Verification - mean: {x_randn.mean():.6f}, var: {x_randn.var():.6f}")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Generator.standard_normal(): {n / average_time:>12,.0f} samples/sec")
    print(f"Legacy np.random.randn():    {n / average_time_randn:>12,.0f} samples/sec")

    if average_time_randn < average_time:
        print(
            f"Legacy randn speedup:        {average_time / average_time_randn:>12.2f}x"
        )
    else:
        print(
            f"Generator speedup:           {average_time_randn / average_time:>12.2f}x"
        )
