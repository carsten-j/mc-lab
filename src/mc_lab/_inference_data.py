"""Shared utilities for creating ArviZ InferenceData objects from MCMC samples.

This module provides common functionality for converting MCMC sampling results
into standardized ArviZ InferenceData objects across different sampling algorithms.
"""

from __future__ import annotations

from typing import Dict, Optional

import arviz as az
import numpy as np
import xarray as xr


def create_inference_data(
    posterior_samples: Dict[str, np.ndarray],
    sample_stats: Dict[str, np.ndarray],
    n_chains: int,
    n_samples: int,
    n_dim: Optional[int] = None,
    algorithm_name: str = "MCMC",
    **metadata,
) -> az.InferenceData:
    """
    Create ArviZ InferenceData object from MCMC sampling results.

    This function standardizes the creation of InferenceData objects across
    different MCMC sampling algorithms, handling common patterns like
    coordinate creation, dataset construction, and metadata assignment.

    Parameters
    ----------
    posterior_samples : Dict[str, np.ndarray]
        Dictionary mapping variable names to posterior sample arrays.
        Each array should have shape (n_chains, n_samples).
    sample_stats : Dict[str, np.ndarray]
        Dictionary mapping statistic names to arrays. Each array should
        have appropriate shape for the statistic type:
        - Scalar stats: (n_chains, n_samples)
        - Vector stats: (n_chains, n_samples, n_dim)
        Can be empty dict if no sample statistics are available.
    n_chains : int
        Number of chains used in sampling.
    n_samples : int
        Number of samples per chain (after burn-in and thinning).
    n_dim : int, optional
        Dimensionality of the problem. Used for creating dimension coordinates
        for multidimensional statistics like proposal_scale. If None, inferred
        from sample statistics when needed.
    algorithm_name : str, default="MCMC"
        Name of the sampling algorithm for metadata.
    **metadata
        Additional metadata to store as attributes on the posterior dataset.
        Common examples include burn_in, thin, step_size, etc.

    Returns
    -------
    idata : arviz.InferenceData
        InferenceData object containing posterior samples and diagnostics.

    Notes
    -----
    **Coordinate Handling:**

    - Always creates "chain" and "draw" coordinates
    - Creates "dim" coordinate when n_dim > 1 or when needed for sample statistics
    - Handles both scalar and vector sample statistics appropriately

    **Sample Statistics Handling:**

    The function automatically handles different types of sample statistics:

    - Scalar statistics (e.g., log_likelihood, accepted): stored as (chain, draw)
    - Vector statistics (e.g., proposal_scale in multidimensional problems):
      stored as (chain, draw, dim) when n_dim > 1, or as (chain, draw) when n_dim = 1

    **Special Cases:**

    - Empty sample_stats dict: Creates InferenceData with only posterior
    - Missing n_dim: Inferred from proposal_scale shape when available
    - Metadata: Stored as attributes on the posterior dataset

    Examples
    --------
    Basic usage with scalar samples:

    >>> posterior_samples = {"x": np.random.randn(4, 1000)}
    >>> sample_stats = {
    ...     "log_likelihood": np.random.randn(4, 1000),
    ...     "accepted": np.random.choice([True, False], (4, 1000))
    ... }
    >>> idata = create_inference_data(
    ...     posterior_samples, sample_stats, n_chains=4, n_samples=1000,
    ...     algorithm_name="Metropolis-Hastings", burn_in=500
    ... )

    Multidimensional case with vector statistics:

    >>> posterior_samples = {
    ...     "x0": np.random.randn(4, 1000),
    ...     "x1": np.random.randn(4, 1000)
    ... }
    >>> sample_stats = {
    ...     "log_likelihood": np.random.randn(4, 1000),
    ...     "proposal_scale": np.random.randn(4, 1000, 2)
    ... }
    >>> idata = create_inference_data(
    ...     posterior_samples, sample_stats, n_chains=4, n_samples=1000,
    ...     n_dim=2, algorithm_name="MALA", step_size=0.1
    ... )
    """
    # Create base coordinates
    coords = {
        "chain": np.arange(n_chains),
        "draw": np.arange(n_samples),
    }

    # Infer n_dim if not provided and needed
    if n_dim is None and "proposal_scale" in sample_stats:
        proposal_scale_shape = sample_stats["proposal_scale"].shape
        if len(proposal_scale_shape) == 3:
            n_dim = proposal_scale_shape[2]
        else:
            n_dim = 1

    # Add dimension coordinate if needed
    if n_dim is not None and n_dim > 1:
        coords["dim"] = np.arange(n_dim)

    # Create posterior dataset
    posterior_dict = {}
    for name, samples in posterior_samples.items():
        posterior_dict[name] = (["chain", "draw"], samples)

    posterior_ds = xr.Dataset(posterior_dict, coords=coords)

    # Create sample_stats dataset if we have statistics
    sample_stats_ds = None
    if sample_stats:
        sample_stats_dict = {}
        for stat_name, values in sample_stats.items():
            if stat_name == "proposal_scale":
                # Handle proposal_scale which can be scalar or vector
                if len(values.shape) == 3:  # (n_chains, n_samples, n_dim)
                    if n_dim == 1:
                        # Extract scalar values for 1D case
                        sample_stats_dict[stat_name] = (
                            ["chain", "draw"],
                            values[:, :, 0],
                        )
                    else:
                        # Keep full dimensionality for multidimensional case
                        sample_stats_dict[stat_name] = (
                            ["chain", "draw", "dim"],
                            values,
                        )
                else:
                    # Already scalar shape (n_chains, n_samples)
                    sample_stats_dict[stat_name] = (["chain", "draw"], values)
            else:
                # Standard scalar statistics
                sample_stats_dict[stat_name] = (["chain", "draw"], values)

        sample_stats_ds = xr.Dataset(sample_stats_dict, coords=coords)

    # Add algorithm name and metadata as attributes
    posterior_attrs = {"sampling_method": algorithm_name}
    posterior_attrs.update(metadata)
    posterior_ds.attrs.update(posterior_attrs)

    # Create InferenceData
    idata = az.InferenceData(
        posterior=posterior_ds,
        sample_stats=sample_stats_ds if sample_stats_ds is not None else None,
    )

    return idata
