"""
Transformation methods (Examples 3.1 and 3.2 from the MCMC notes).

Example 3.1 (Exponential via inverse CDF):
        If U ~ Uniform(0,1), then X = F^{-1}(U) = -ln(1-U)/lambda ~ Exponential(rate=lambda).

Example 3.2 (Cauchy via tangent transform):
        If U ~ Uniform(0,1), then X = tan(pi (U - 1/2)) ~ Cauchy(location=0, scale=1).
        More generally, X = loc + scale * tan(pi (U - 1/2)).

Both routines below produce i.i.d. samples using these transforms.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ._rng import RandomState, as_generator

__all__ = [
    "sample_exponential_via_inverse",
    "sample_cauchy_via_tangent",
]


def _deprecated_local_as_generator(random_state: RandomState) -> np.random.Generator:
    # Back-compat shim in case internal references exist; prefer as_generator
    return as_generator(random_state)


def sample_exponential_via_inverse(
    n: int,
    rate: float = 1.0,
    random_state: RandomState = None,
    *,
    dtype: Union[np.dtype, str] = np.float64,
) -> np.ndarray:
    """Example 3.1: Exponential sampling via inverse CDF.

    Generate X ~ Exp(rate) from U ~ Uniform(0,1) using X = -ln(1-U)/rate.

    Parameters
    ----------
    n : int
            Number of samples.
    rate : float
            Rate parameter (lambda > 0). Mean = 1/rate.
    random_state : int | numpy.random.Generator | None
            RNG seed or Generator for reproducibility.
    dtype : numpy dtype
            Floating dtype of the output (default float64).

    Returns
    -------
    np.ndarray, shape (n,)
            Exponential samples.
    """
    if n <= 0:
        return np.empty(0, dtype=dtype)
    if rate <= 0:
        raise ValueError("rate must be positive")

    rng = as_generator(random_state)

    # Draw U in (0,1). Guard against U=0 => log(1-0)=0 safe; prefer open interval.
    # We use np.nextafter to avoid exactly 0 or 1 when dtype is finite precision.
    u = rng.random(n, dtype=dtype)
    eps0 = np.nextafter(np.array(0, dtype=dtype), np.array(1, dtype=dtype))
    eps1 = np.nextafter(np.array(1, dtype=dtype), np.array(0, dtype=dtype))
    u = np.clip(u, eps0, eps1)

    x = -np.log1p(-u) / rate  # log1p for better accuracy when u is small
    return x.astype(dtype, copy=False)


def sample_cauchy_via_tangent(
    n: int,
    location: float = 0.0,
    scale: float = 1.0,
    random_state: RandomState = None,
    *,
    dtype: Union[np.dtype, str] = np.float64,
) -> np.ndarray:
    """Example 3.2: Cauchy sampling via tangent transform.

    Generate X ~ Cauchy(loc, scale) from U ~ Uniform(0,1) using
    X = loc + scale * tan(pi (U - 1/2)).

    Parameters
    ----------
    n : int
            Number of samples.
    location : float
            Location parameter (median).
    scale : float
            Scale parameter (> 0) controlling half-width at half-maximum.
    random_state : int | numpy.random.Generator | None
            RNG seed or Generator for reproducibility.
    dtype : numpy dtype
            Floating dtype of the output (default float64).

    Returns
    -------
    np.ndarray, shape (n,)
            Cauchy samples.
    """
    if n <= 0:
        return np.empty(0, dtype=dtype)
    if scale <= 0:
        raise ValueError("scale must be positive")

    rng = as_generator(random_state)
    u = rng.random(n, dtype=dtype)

    # Map U from (0,1) to angle in (-pi/2, pi/2) while avoiding exact endpoints
    # to prevent infinite tan. Clip slightly away from 0 and 1.
    eps0 = np.nextafter(np.array(0, dtype=dtype), np.array(1, dtype=dtype))
    eps1 = np.nextafter(np.array(1, dtype=dtype), np.array(0, dtype=dtype))
    u = np.clip(u, eps0, eps1)

    angles = np.pi * (u - 0.5)
    x = location + scale * np.tan(angles)
    return x.astype(dtype, copy=False)
