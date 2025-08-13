"""
Fast and efficient Box–Muller normal random variate generators.

Two vectorized implementations are provided:
- Classic Box–Muller using sin/cos on two independent U(0,1) variates.
- Marsaglia's polar method (rejection sampling) avoiding trig for speed.

Both return standard normal samples N(0,1). Use the optional random_state
to control randomness (int seed or numpy.random.Generator).
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np

Method = Literal["classic", "polar"]

__all__ = [
    "box_muller",
    "box_muller_pairs",
]


def _as_generator(
    random_state: Optional[Union[int, np.random.Generator]],
) -> np.random.Generator:
    """Return a NumPy Generator from an int seed, Generator, or None.

    None -> default bit generator.
    int  -> new PCG64 generator with the given seed.
    Generator -> returned as-is.
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is None:
        return np.random.default_rng()
    return np.random.default_rng(random_state)


def box_muller(
    n: int,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    method: Method = "classic",
    return_pairs: bool = False,
    dtype: Union[np.dtype, str] = np.float64,
) -> np.ndarray:
    """Generate standard normal samples using the Box–Muller transform.

    Parameters
    ----------
    n : int
            Number of samples to generate.
    random_state : int | numpy.random.Generator | None
            Seed or Generator for reproducibility. If None, uses a fresh default RNG.
    method : {"classic", "polar"}
            Implementation to use. "classic" uses sin/cos; "polar" avoids trig with
            Marsaglia's acceptance-rejection method.
    return_pairs : bool
            If True, return shape (m, 2) of independent normals (m = ceil(n/2)).
            If False, return a flat array of length n.
    dtype : numpy dtype
            Floating dtype of the output (default float64).

    Returns
    -------
    np.ndarray
            Array of N(0,1) samples. Shape (n,) by default or (m,2) if return_pairs.
    """
    if n <= 0:
        return (
            np.empty((0, 2), dtype=dtype) if return_pairs else np.empty(0, dtype=dtype)
        )

    rng = _as_generator(random_state)

    if method == "classic":
        samples = _box_muller_classic(n, rng, dtype=dtype, return_pairs=return_pairs)
    elif method == "polar":
        samples = _box_muller_polar(n, rng, dtype=dtype, return_pairs=return_pairs)
    else:
        raise ValueError("method must be 'classic' or 'polar'")

    return samples


def box_muller_pairs(
    m: int,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    method: Method = "classic",
    dtype: Union[np.dtype, str] = np.float64,
) -> np.ndarray:
    """Generate m pairs of independent standard normals using Box–Muller.

    Parameters
    ----------
    m : int
            Number of independent pairs to generate.
    random_state : int | numpy.random.Generator | None
            Seed or Generator for reproducibility.
    method : {"classic", "polar"}
            Implementation to use.
    dtype : numpy dtype
            Floating dtype of the output (default float64).

    Returns
    -------
    np.ndarray
            Array of shape (m, 2) with i.i.d. N(0,1) entries.
    """
    return box_muller(
        n=2 * m,
        random_state=random_state,
        method=method,
        return_pairs=True,
        dtype=dtype,
    )


def _box_muller_classic(
    n: int,
    rng: np.random.Generator,
    *,
    dtype: Union[np.dtype, str] = np.float64,
    return_pairs: bool = False,
) -> np.ndarray:
    """Vectorized classic Box–Muller using two U(0,1) variates and sin/cos."""
    m = (n + 1) // 2  # number of pairs needed
    # Draw uniforms and guard against log(0)
    u1 = rng.random(m, dtype=dtype)
    # ensure u1 in (0,1]; clip away from 0 to avoid -inf
    tiny = np.finfo(np.dtype(dtype)).tiny
    u1 = np.maximum(u1, tiny)
    u2 = rng.random(m, dtype=dtype)

    # Cast operands to ensure desired dtype, ufuncs will follow input dtype
    u1 = u1.astype(dtype, copy=False)
    u2 = u2.astype(dtype, copy=False)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = (2.0 * np.pi * u2).astype(dtype, copy=False)

    z0 = r * np.cos(theta)
    z1 = r * np.sin(theta)

    if return_pairs:
        out = np.empty((m, 2), dtype=dtype)
        out[:, 0] = z0
        out[:, 1] = z1
        return out

    # Flat array of length n
    out = np.empty(2 * m, dtype=dtype)
    out[0::2] = z0
    out[1::2] = z1
    return out[:n]


def _box_muller_polar(
    n: int,
    rng: np.random.Generator,
    *,
    dtype: Union[np.dtype, str] = np.float64,
    return_pairs: bool = False,
) -> np.ndarray:
    """Marsaglia's polar method (rejection sampling) avoiding trig.

    Generates pairs (Z1,Z2) ~ N(0,1) i.i.d. using:
    - Draw U,V ~ Uniform(-1,1)
    - s = U^2 + V^2; accept if 0 < s < 1
    - factor = sqrt(-2 ln s / s)
    - Z1 = U * factor, Z2 = V * factor
    """
    target_pairs = (n + 1) // 2
    acc_u: list[np.ndarray] = []
    acc_v: list[np.ndarray] = []

    # Generate in chunks until we have enough accepted pairs
    remaining = target_pairs
    while remaining > 0:
        # Over-sample a bit to reduce iterations; expected acceptance ~ pi/4 ~ 0.785
        chunk = max(remaining * 2, 1024)
        u = rng.uniform(-1.0, 1.0, size=chunk).astype(dtype, copy=False)
        v = rng.uniform(-1.0, 1.0, size=chunk).astype(dtype, copy=False)
        s = u * u + v * v
        mask = (s > 0.0) & (s < 1.0)

        if not np.any(mask):
            continue

        u = u[mask]
        v = v[mask]
        s = s[mask]

        # factor = sqrt(-2 * ln(s) / s)
        s = s.astype(dtype, copy=False)
        factor = np.sqrt(-2.0 * np.log(s) / s)
        acc_u.append(u * factor)
        acc_v.append(v * factor)

        remaining -= u.shape[0]

    z0 = np.concatenate(acc_u, axis=0)[:target_pairs]
    z1 = np.concatenate(acc_v, axis=0)[:target_pairs]

    if return_pairs:
        out = np.empty((target_pairs, 2), dtype=dtype)
        out[:, 0] = z0
        out[:, 1] = z1
        return out

    out = np.empty(2 * target_pairs, dtype=dtype)
    out[0::2] = z0
    out[1::2] = z1
    return out[:n]
