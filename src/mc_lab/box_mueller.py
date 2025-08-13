"""
Fast and efficient Box-Muller normal random variate generators.

Two vectorized implementations are provided:
- Classic Box-Muller using sin/cos on two independent U(0,1) variates.
- Marsaglia's polar method (rejection sampling) avoiding trig for speed.

Both return standard normal samples N(0,1). Use the optional random_state
to control randomness (int seed or numpy.random.Generator).

References
----------
- G. E. P. Box and M. E. Muller (1958). "A Note on the Generation of Random
    Normal Deviates." The Annals of Mathematical Statistics, 29(2): 610-611.
    doi:10.1214/aoms/1177706645.
- G. Marsaglia and T. A. Bray (1964). "A Convenient Method for Generating
    Normal Variables." SIAM Review, 6(3): 260-264. doi:10.1137/1006063.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._rng import RandomState, RNGLike, as_generator

Method = Literal["classic", "polar"]

__all__ = ["box_muller"]


def box_muller(
    n: int,
    random_state: RandomState = None,
    method: Method = "classic",
    return_pairs: bool = False,
) -> np.ndarray:
    """Generate standard normal samples using the Box-Muller transform.

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
    Notes
    -----
    - Output dtype is always float64.

    Returns
    -------
    np.ndarray
            Array of N(0,1) samples. Shape (n,) by default or (m,2) if return_pairs.
    """
    if n <= 0:
        return (
            np.empty((0, 2), dtype=np.float64)
            if return_pairs
            else np.empty(0, dtype=np.float64)
        )

    rng = as_generator(random_state)

    if method == "classic":
        samples = _box_muller_classic(n, rng, return_pairs=return_pairs)
    elif method == "polar":
        samples = _box_muller_polar(n, rng, return_pairs=return_pairs)
    else:
        raise ValueError("method must be 'classic' or 'polar'")

    return samples


def _box_muller_classic(
    n: int,
    rng: RNGLike,
    *,
    return_pairs: bool = False,
) -> np.ndarray:
    """Vectorized classic Box-Muller using two U(0,1) variates and sin/cos."""
    m = (n + 1) // 2  # number of pairs needed
    # Draw uniforms and guard against log(0)
    u1 = rng.random(m, dtype=np.float64)
    # ensure u1 in (0,1]; clip away from 0 to avoid -inf
    tiny = np.finfo(np.float64).tiny
    u1 = np.maximum(u1, tiny)
    u2 = rng.random(m, dtype=np.float64)

    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2

    z0 = r * np.cos(theta)
    z1 = r * np.sin(theta)

    return _assemble_output(z0, z1, n=n, return_pairs=return_pairs)


def _box_muller_polar(
    n: int,
    rng: RNGLike,
    *,
    return_pairs: bool = False,
) -> np.ndarray:
    """Marsaglia's polar method (rejection sampling) avoiding trig. functions

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
        u = rng.uniform(-1.0, 1.0, size=chunk)
        v = rng.uniform(-1.0, 1.0, size=chunk)
        s = u * u + v * v
        mask = (s > 0.0) & (s < 1.0)

        if not np.any(mask):
            continue

        u = u[mask]
        v = v[mask]
        s = s[mask]

        # factor = sqrt(-2 * ln(s) / s)
        factor = np.sqrt(-2.0 * np.log(s) / s)
        acc_u.append(u * factor)
        acc_v.append(v * factor)

        remaining -= u.shape[0]

    z0 = np.concatenate(acc_u, axis=0)[:target_pairs]
    z1 = np.concatenate(acc_v, axis=0)[:target_pairs]

    return _assemble_output(z0, z1, n=n, return_pairs=return_pairs)


def _assemble_output(
    z0: np.ndarray, z1: np.ndarray, *, n: int, return_pairs: bool
) -> np.ndarray:
    """Assemble output either as pairs (m,2) or flat (n,) from two vectors.

    Assumes z0 and z1 have equal length m >= ceil(n/2).
    """
    m = z0.shape[0]
    if return_pairs:
        out = np.empty((m, 2), dtype=np.float64)
        out[:, 0] = z0
        out[:, 1] = z1
        return out

    out = np.empty(2 * m, dtype=np.float64)
    out[0::2] = z0
    out[1::2] = z1
    return out[:n]
