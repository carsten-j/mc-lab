"""
Multivariate Gaussian random sampler (Example 3.4, MCMC notes).

Core idea:
- Draw Z ~ N(0, I_d)
- Factorize covariance Sigma = L L^T (Cholesky) and set X = mu + L Z

If Sigma is not numerically positive definite, fall back to an eigen
decomposition with non-negative eigen clipping.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ._rng import RandomState, as_generator

__all__ = ["sample_multivariate_gaussian"]


def _deprecated_local_as_generator(random_state: RandomState) -> np.random.Generator:
    # Back-compat shim in case internal references exist; prefer as_generator
    return as_generator(random_state)


def _factor_from_cov(
    cov: np.ndarray, method: str = "cholesky", jitter: float = 0.0
) -> np.ndarray:
    """Return matrix A s.t. A @ A.T = cov.

    method="cholesky": use np.linalg.cholesky; optionally add jitter*I on failure.
    method="eigh": symmetric eigen-decomposition with non-negative clipping.
    """
    cov = np.asarray(cov)
    d = cov.shape[0]
    if method == "cholesky":
        try:
            return np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Try a small jitter and retry once
            if jitter == 0.0:
                jitter = max(1e-12, 1e-12 * np.trace(cov) / max(d, 1))
            try:
                return np.linalg.cholesky(cov + jitter * np.eye(d, dtype=cov.dtype))
            except np.linalg.LinAlgError:
                # Fall back to eigen
                method = "eigh"

    if method == "eigh":
        # Force symmetry and eigendecompose
        sym = 0.5 * (cov + cov.T)
        w, Q = np.linalg.eigh(sym)
        # Clip tiny negatives due to numerical issues
        w_clipped = np.clip(w, a_min=0.0, a_max=None)
        # Construct factor A = Q diag(sqrt(w)) so that A A^T = sym
        A = Q @ np.diag(np.sqrt(w_clipped))
        return A

    raise ValueError("method must be 'cholesky' or 'eigh'")


def sample_multivariate_gaussian(
    mean: Union[np.ndarray, list, tuple],
    cov: Union[np.ndarray, list],
    n: int,
    random_state: RandomState = None,
    *,
    method: str = "cholesky",
    dtype: Union[np.dtype, str] = np.float64,
) -> np.ndarray:
    """Generate samples X ~ N(mean, cov) using linear transform of standard normals.

    Parameters
    ----------
    mean : array-like, shape (d,)
            Mean vector.
    cov : array-like, shape (d, d)
            Symmetric positive semi-definite covariance matrix.
    n : int
            Number of samples to draw.
    random_state : int | numpy.random.Generator | None
            RNG seed or Generator for reproducibility.
    method : {"cholesky", "eigh"}
            Factorization method. "cholesky" is fastest when cov is PD; will fall back
            to eigen if Cholesky fails. "eigh" uses eigen-decomposition with clipping.
    dtype : numpy dtype
            Floating dtype of returned samples.

    Returns
    -------
    np.ndarray, shape (n, d)
            Samples with desired mean and covariance.
    """
    if n <= 0:
        mean_arr = np.asarray(mean, dtype=dtype)
        d = int(mean_arr.shape[-1])
        return np.empty((0, d), dtype=dtype)

    rng = as_generator(random_state)

    mean_arr = np.asarray(mean, dtype=dtype)
    cov_arr = np.asarray(cov, dtype=dtype)
    if mean_arr.ndim != 1:
        raise ValueError("mean must be a 1D array-like of shape (d,)")
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValueError("cov must be a square 2D array of shape (d, d)")
    d = mean_arr.shape[0]
    if cov_arr.shape != (d, d):
        raise ValueError("cov shape must match mean dimension")

    # Obtain factor A such that A A^T = cov
    A = _factor_from_cov(cov_arr, method=method)

    # Standard normal draws (n, d)
    Z = rng.standard_normal(size=(n, d)).astype(dtype, copy=False)

    # Transform: X = mu + Z @ A^T
    X = Z @ A.T
    X += mean_arr  # broadcasts over rows
    return X.astype(dtype, copy=False)
