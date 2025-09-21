#!/usr/bin/env python3
import argparse
import gc
import math
import platform
import time
from typing import Optional, Tuple, Union

import numpy as np
from numpy.random import Generator, default_rng

# Optional SciPy import (required for the SciPy baseline and KS test)
try:
    import scipy
    from scipy import stats

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ---------------------------
# Jöhnk-based Gamma(a, 1)
# ---------------------------


def _gamma_johnk_frac(a: float, n: int, rng: Generator, batch: int) -> np.ndarray:
    """
    Generate n i.i.d. Gamma(a, 1) for 0 < a < 1 using Jöhnk's wedge method.
    Fixed to cap the number of accepted samples per loop to the remaining capacity.
    """
    if not (0.0 < a < 1.0):
        raise ValueError("Internal: _gamma_johnk_frac requires 0 < a < 1.")
    out = np.empty(n, dtype=np.float64)
    filled = 0

    inv_a = 1.0 / a
    inv_1ma = 1.0 / (1.0 - a)
    batch = max(1024, int(batch))

    while filled < n:
        need = n - filled
        m = max(
            batch, need
        )  # produce at least a batch, but no smaller than what we still need
        u = rng.random(m, dtype=np.float64)
        v = rng.random(m, dtype=np.float64)
        x = u**inv_a
        y = v**inv_1ma
        s = x + y

        acc_idx = np.flatnonzero(s <= 1.0)
        if acc_idx.size == 0:
            continue

        k = min(need, acc_idx.size)  # do not overfill
        sel = acc_idx[:k]
        t = x[sel] / s[sel]  # Beta(a, 1-a)
        e = rng.exponential(1.0, size=k)  # Exp(1)
        out[filled : filled + k] = e * t  # Gamma(a, 1)
        filled += k

    return out


def _gamma_integer_sum(
    k: int, n: int, rng: Generator, max_exp_elems: int = 1_000_000
) -> np.ndarray:
    """
    Generate n i.i.d. Gamma(k,1) for integer k >= 0 by summing exponentials, in chunks.
    """
    if k < 0:
        raise ValueError("Internal: _gamma_integer_sum requires k >= 0.")
    if k == 0:
        return np.zeros(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    # Each chunk materializes at most ~max_exp_elems exponentials.
    chunk_rows = max(1, int(max_exp_elems // max(1, k)))
    pos = 0
    while pos < n:
        m = min(chunk_rows, n - pos)
        out[pos : pos + m] = rng.exponential(1.0, size=(m, k)).sum(axis=1)
        pos += m
    return out


def gamma_johnk(
    a: float,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    rng: Optional[Generator] = None,
    *,
    batch: int = 65536,
    max_exp_elems: int = 1_000_000,
):
    """
    Gamma(a, 1) using Jöhnk for fractional part (0<a<1); integer part via sum of exponentials.
    """
    if rng is None:
        rng = default_rng()
    a = float(a)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("a must be a positive finite scalar.")

    # Resolve output shape and flat sample count
    if size is None:
        out_shape = ()
        n = 1
    else:
        out_shape = (size,) if isinstance(size, int) else tuple(size)
        n = int(np.prod(out_shape)) if out_shape else 1

    if np.isclose(a, 1.0):
        res = rng.exponential(1.0, size=n)
    elif a < 1.0:
        res = _gamma_johnk_frac(a, n, rng, batch)
    else:
        k = int(np.floor(a))
        r = a - k
        gk = _gamma_integer_sum(k, n, rng, max_exp_elems=max_exp_elems)
        if r > 0.0 and not np.isclose(r, 0.0):
            gr = _gamma_johnk_frac(r, n, rng, batch)
            res = gk + gr
        else:
            res = gk
    return res.reshape(out_shape) if size is not None else float(res[0])


# ---------------------------
# Benchmark utilities
# ---------------------------


def bench_callable(call, repeats: int = 3) -> Tuple[float, float]:
    """
    Time a zero-arg callable over `repeats` runs.
    Returns (avg_seconds, min_seconds).
    """
    times = []
    # Warm-up
    call()
    gcold = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            call()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    finally:
        if gcold:
            gc.enable()
    return (sum(times) / len(times), min(times))


def summarize_accuracy(x: np.ndarray, a: float) -> dict:
    """
    Simple accuracy summary: sample mean/var, relative error versus theory (mean=var=a).
    """
    m = float(np.mean(x))
    v = float(np.var(x))
    return {
        "mean": m,
        "var": v,
        "rel_err_mean": (m - a) / a,
        "rel_err_var": (v - a) / a,
    }


def ks_test(x: np.ndarray, a: float) -> Tuple[float, float]:
    """
    KS test against Gamma(a,1) CDF (requires SciPy). Returns (D, pvalue).
    """
    if not SCIPY_AVAILABLE:
        return (math.nan, math.nan)
    D, p = stats.kstest(x, stats.gamma(a, scale=1.0).cdf)
    return (float(D), float(p))


def format_float(x, width=9, prec=3):
    return f"{x:{width}.{prec}f}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark Jöhnk Gamma(a,1) vs SciPy")
    parser.add_argument(
        "-a",
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.9, 1.0, 1.5, 2.7, 5.3, 10.0],
        help="Shape parameters a to test.",
    )
    parser.add_argument(
        "-n",
        "--nsamples",
        type=int,
        default=500_000,
        help="Number of samples per a per run.",
    )
    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=3,
        help="Number of timing repeats to average.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Master RNG seed used to seed each method per run.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=65536,
        help="Batch size for Jöhnk accept-reject (performance knob).",
    )
    parser.add_argument(
        "--max-exp-elems",
        type=int,
        default=1_000_000,
        help="Memory budget for integer-part exponentials (rows*shape <= this).",
    )
    parser.add_argument(
        "--include-numpy",
        action="store_true",
        help="Also benchmark NumPy's Generator.gamma as a second baseline.",
    )
    parser.add_argument(
        "--do-ks",
        action="store_true",
        help="Run a KS test on 100k subsample (requires SciPy).",
    )
    args = parser.parse_args()

    print("Environment:")
    print(f"- Python: {platform.python_version()}")
    print(f"- NumPy:  {np.__version__}")
    print(f"- SciPy:  {scipy.__version__ if SCIPY_AVAILABLE else 'not available'}")
    print(
        f"- Machine: {platform.processor() or platform.machine()}  ({platform.system()} {platform.release()})"
    )
    print()
    print(
        f"Config: nsamples={args.nsamples:,}, repeats={args.repeats}, seed={args.seed}, batch={args.batch}, max_exp_elems={args.max_exp_elems}"
    )
    if not SCIPY_AVAILABLE:
        print(
            "WARNING: SciPy not available; SciPy baseline and KS test will be skipped."
        )
    print()

    header_cols = [
        "a",
        "N",
        "Johnk s",
        "Johnk Ms/s",
        "SciPy s",
        "SciPy Ms/s",
        "Speedup(J/S)",
        "mean err%",
        "var err%",
    ]
    if args.include_numpy:
        header_cols[4:4] = ["NumPy s", "NumPy Ms/s", "Speedup(J/N)"]
    if args.do_ks and SCIPY_AVAILABLE:
        header_cols += ["KS D", "KS pval"]

    print(" | ".join(header_cols))

    for a in args.alphas:
        N = int(args.nsamples)

        # Build callables that close over fresh RNGs each timing run
        def johnk_once():
            rng = default_rng(args.seed)  # re-seed for fairness across repeats
            return gamma_johnk(
                a, size=N, rng=rng, batch=args.batch, max_exp_elems=args.max_exp_elems
            )

        if SCIPY_AVAILABLE:

            def scipy_once():
                rng = default_rng(args.seed)
                # scipy.stats.gamma.rvs supports numpy Generator as random_state
                return stats.gamma.rvs(a, scale=1.0, size=N, random_state=rng)
        else:
            scipy_once = None

        if args.include_numpy:

            def numpy_once():
                rng = default_rng(args.seed)
                return rng.gamma(shape=a, scale=1.0, size=N)
        else:
            numpy_once = None

        # Time Jöhnk
        _, j_min = bench_callable(johnk_once, repeats=args.repeats)
        j_thru = (N / j_min) / 1e6

        # Time NumPy baseline (optional)
        if numpy_once is not None:
            n_avg, n_min = bench_callable(numpy_once, repeats=args.repeats)
            n_thru = (N / n_min) / 1e6
        else:
            _ = n_min = n_thru = math.nan

        # Time SciPy baseline
        if scipy_once is not None:
            _, s_min = bench_callable(scipy_once, repeats=args.repeats)
            s_thru = (N / s_min) / 1e6
        else:
            _ = s_min = s_thru = math.nan

        # Accuracy on a single Jöhnk draw
        x = johnk_once()
        acc = summarize_accuracy(x, a)

        # KS test (optional; subsample to keep it quick)
        if args.do_ks and SCIPY_AVAILABLE:
            subsz = min(100_000, N)
            idx = default_rng(args.seed).integers(0, N, size=subsz, endpoint=False)
            D, pval = ks_test(x[idx], a)
        else:
            D = pval = math.nan

        # Prepare row
        row = [
            format_float(a, width=5, prec=2),
            f"{N:,}",
            format_float(j_min, prec=3),
            format_float(j_thru, prec=2),
        ]
        if args.include_numpy:
            row += [
                format_float(n_min, prec=3),
                format_float(n_thru, prec=2),
                format_float(n_min / j_min, prec=2),
            ]
        row += [
            format_float(s_min, prec=3),
            format_float(s_thru, prec=2),
            format_float(s_min / j_min, prec=2) if np.isfinite(s_min) else "   nan",
            format_float(100 * acc["rel_err_mean"], prec=2),
            format_float(100 * acc["rel_err_var"], prec=2),
        ]
        if args.do_ks and SCIPY_AVAILABLE:
            row += [format_float(D, prec=3), f"{pval:.2e}"]
        print(" | ".join(row))


if __name__ == "__main__":
    main()
