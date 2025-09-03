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

# # Optional Numba acceleration for the polar method
# try:  # pragma: no cover - availability depends on environment
#     import numba as _numba  # type: ignore

#     _HAVE_NUMBA = True
# except Exception:  # pragma: no cover
#     _numba = None
#     _HAVE_NUMBA = False

# Method = Literal["classic", "polar", "polar_njit"]
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
    method : {"classic", "polar", "polar_njit"}
            Implementation to use. "classic" uses sin/cos; "polar" avoids trig with
            Marsaglia's acceptance-rejection method. "polar_njit" uses a
            Numba-accelerated rejection loop (if numba is installed).
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
    # elif method == "polar_njit":
    #     samples = _box_muller_polar_njit(n, rng, return_pairs=return_pairs)
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
    # Draw both uniforms in a single call to reduce RNG call overhead
    u = rng.random((2, m), dtype=np.float64)
    u1 = u[0]
    u2 = u[1]

    # Guard against log(0) by bumping zeros up to tiny (probability of zero is ~2^-53)
    tiny = np.finfo(np.float64).tiny
    np.maximum(u1, tiny, out=u1)

    # r = sqrt(-2 * log(u1)) computed in-place to reuse u1's memory
    np.log(u1, out=u1)  # u1 := log(u1)
    u1 *= -2.0
    np.sqrt(u1, out=u1)  # u1 := r

    # theta = 2*pi*u2, computed in-place
    u2 *= 2.0 * np.pi

    if return_pairs:
        out = np.empty((m, 2), dtype=np.float64)
        # out[:,0] = r * cos(theta); out[:,1] = r * sin(theta)
        col0 = out[:, 0]
        col1 = out[:, 1]
        np.cos(u2, out=col0)
        col0 *= u1
        np.sin(u2, out=col1)
        col1 *= u1
        return out

    # Interleave directly into the output buffer to avoid temporaries
    out = np.empty(2 * m, dtype=np.float64)
    even = out[0::2]
    odd = out[1::2]
    np.cos(u2, out=even)
    even *= u1
    np.sin(u2, out=odd)
    odd *= u1
    return out[:n]


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
    # Preallocate outputs to avoid list growth and concatenations
    z0 = np.empty(target_pairs, dtype=np.float64)
    z1 = np.empty(target_pairs, dtype=np.float64)

    remaining = target_pairs
    pos = 0
    # Generate in chunks until we have enough accepted pairs
    while remaining > 0:
        # Choose chunk size so that expected accepts cover remaining in ~1 iteration.
        # Acceptance rate ~ pi/4 ≈ 0.785.
        chunk = max(int(remaining / 0.785) + 1, 1024)
        uv = rng.uniform(-1.0, 1.0, size=(2, chunk))
        u = uv[0]
        v = uv[1]
        s = u * u + v * v
        mask = (s > 0.0) & (s < 1.0)
        if not np.any(mask):
            continue

        u = u[mask]
        v = v[mask]
        s = s[mask]

        # factor = sqrt(-2 * ln(s) / s)
        factor = np.sqrt(-2.0 * np.log(s) / s)

        k = min(u.shape[0], remaining)
        # Multiply in-place on the masked views to avoid temporaries beyond 'factor'
        z0[pos : pos + k] = u[:k] * factor[:k]
        z1[pos : pos + k] = v[:k] * factor[:k]
        pos += k
        remaining -= k

    return _assemble_output(z0, z1, n=n, return_pairs=return_pairs)


# def _box_muller_polar_njit(
#     n: int,
#     rng: RNGLike,
#     *,
#     return_pairs: bool = False,
# ) -> np.ndarray:
#     """Numba-accelerated Marsaglia polar method.

#     Uses a compiled scalar rejection loop with its own RNG seeded from the
#     provided random_state for reproducibility across runs.
#     """
#     if not _HAVE_NUMBA:
#         raise ImportError(
#             "Numba is required for method='polar_njit'. Please install 'numba' or use method='polar'."
#         )

#     target_pairs = (n + 1) // 2

#     # Derive an integer seed from rng so results are reproducible given the input state
#     # We avoid consuming many values from rng; just one 32-bit value for seeding.
#     try:
#         seed = int(
#             np.uint64(
#                 np.random.SeedSequence(
#                     rng.integers(0, 2**32 - 1, dtype=np.uint32)
#                 ).entropy
#             )
#         )
#     except Exception:
#         # Fallback if rng is not a numpy Generator-like object with integers
#         seed = int(np.random.SeedSequence().entropy)

#     z0, z1 = _polar_pairs_numba(target_pairs, seed)
#     return _assemble_output(z0, z1, n=n, return_pairs=return_pairs)


# if _HAVE_NUMBA:  # pragma: no cover

#     @_numba.njit(cache=True, fastmath=True)
#     def _polar_pairs_numba(m: int, seed: int):  # type: ignore[no-redef]
#         # Local RNG seeded for reproducibility
#         if seed >= 0:
#             np.random.seed(np.uint64(seed))

#         out0 = np.empty(m, np.float64)
#         out1 = np.empty(m, np.float64)
#         i = 0
#         # Scalar rejection loop; typically ~0.785 acceptance rate
#         while i < m:
#             u = 2.0 * np.random.random() - 1.0
#             v = 2.0 * np.random.random() - 1.0
#             s = u * u + v * v
#             if s > 0.0 and s < 1.0:
#                 factor = np.sqrt(-2.0 * np.log(s) / s)
#                 out0[i] = u * factor
#                 out1[i] = v * factor
#                 i += 1
#         return out0, out1
# else:  # pragma: no cover

#     def _polar_pairs_numba(m: int, seed: int):  # type: ignore[misc]
#         raise ImportError("numba is not available")


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
