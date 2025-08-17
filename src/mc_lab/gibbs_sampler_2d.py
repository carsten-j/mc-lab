from collections.abc import Callable
from typing import Dict, Optional, Tuple

import arviz as az
import numpy as np
import xarray as xr
from tqdm.auto import tqdm


class GibbsSampler2D:
    """
    Gibbs sampler for 2D joint distributions p(x,y) with ArviZ integration.
    Samples from the joint distribution using conditional distributions p(x|y) and p(y|x).
    Returns results as ArviZ InferenceData for advanced diagnostics and visualization.
    """

    def __init__(
        self,
        sample_x_given_y: Callable[[float], float],
        sample_y_given_x: Callable[[float], float],
        log_prob: Optional[Callable[[float, float], float]] = None,
        var_names: Tuple[str, str] = ("x", "y"),
    ):
        """
        Initialize the Gibbs sampler.

        Parameters:
        -----------
        sample_x_given_y : callable
            Function that samples from p(x|y). Takes y value, returns sampled x.
        sample_y_given_x : callable
            Function that samples from p(y|x). Takes x value, returns sampled y.
        log_prob : callable, optional
            Function that computes log p(x,y). Used for diagnostics.
        var_names : tuple of str
            Names for the two variables (default: 'x', 'y')
        """
        self.sample_x_given_y = sample_x_given_y
        self.sample_y_given_x = sample_y_given_x
        self.log_prob = log_prob
        self.var_names = var_names

    def sample(
        self,
        n_samples: int = 1000,
        n_chains: int = 4,
        burn_in: int = 1000,
        thin: int = 1,
        initial_state: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
    ) -> az.InferenceData:
        """
        Generate samples using Gibbs sampling and return as ArviZ InferenceData.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate per chain (after burn-in and thinning).
        n_chains : int
            Number of independent chains to run.
        burn_in : int
            Number of initial samples to discard per chain.
        thin : int
            Keep every 'thin'-th sample to reduce autocorrelation.
        initial_state : np.ndarray, optional
            Initial (x, y) values. Shape should be (n_chains, 2) or (2,).
            If (2,), the same initial state is used for all chains with small random perturbation.
        random_seed : int, optional
            Random seed for reproducibility.
        progressbar : bool
            Show progress bar during sampling.

        Returns:
        --------
        idata : arviz.InferenceData
            InferenceData object containing posterior samples and diagnostics.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Setup initial states for all chains
        if initial_state is None:
            initial_states = np.random.randn(n_chains, 2)
        elif initial_state.shape == (2,):
            # Add small random perturbation to avoid identical chains
            initial_states = (
                initial_state[np.newaxis, :] + np.random.randn(n_chains, 2) * 0.1
            )
        else:
            initial_states = initial_state

        # Storage for all chains
        total_iterations = burn_in + n_samples * thin
        posterior_samples = {
            self.var_names[0]: np.zeros((n_chains, n_samples)),
            self.var_names[1]: np.zeros((n_chains, n_samples)),
        }

        # Sample statistics storage
        sample_stats = {}
        if self.log_prob is not None:
            sample_stats["log_likelihood"] = np.zeros((n_chains, n_samples))
            sample_stats["lp"] = np.zeros((n_chains, n_samples))  # log posterior

        # Run each chain
        for chain_idx in range(n_chains):
            current_state = initial_states[chain_idx].copy()
            sample_idx = 0

            pbar = None
            if progressbar:
                pbar = tqdm(
                    total=total_iterations,
                    desc=f"Chain {chain_idx + 1}/{n_chains}",
                    leave=True,
                    dynamic_ncols=True,
                )

            for iteration in range(total_iterations):
                # Gibbs sampling steps
                current_state[0] = self.sample_x_given_y(current_state[1])
                current_state[1] = self.sample_y_given_x(current_state[0])

                # Store sample if past burn-in and at thinning interval
                if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                    posterior_samples[self.var_names[0]][chain_idx, sample_idx] = (
                        current_state[0]
                    )
                    posterior_samples[self.var_names[1]][chain_idx, sample_idx] = (
                        current_state[1]
                    )

                    # Compute sample statistics if log_prob is provided
                    if self.log_prob is not None:
                        lp = self.log_prob(current_state[0], current_state[1])
                        sample_stats["log_likelihood"][chain_idx, sample_idx] = lp
                        sample_stats["lp"][chain_idx, sample_idx] = lp

                    sample_idx += 1

                    # Update tqdm postfix with collected samples
                    if pbar is not None:
                        pbar.set_postfix(samples=sample_idx)

                # Step progress bar forward
                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

        # Create ArviZ InferenceData object
        idata = self._create_inference_data(
            posterior_samples,
            sample_stats,
            n_chains=n_chains,
            n_samples=n_samples,
            burn_in=burn_in,
            thin=thin,
        )

        return idata

    def _create_inference_data(
        self,
        posterior_samples: Dict[str, np.ndarray],
        sample_stats: Dict[str, np.ndarray],
        n_chains: int,
        n_samples: int,
        burn_in: int,
        thin: int,
    ) -> az.InferenceData:
        """
        Create ArviZ InferenceData object from samples.
        """
        # Create coordinates
        coords = {
            "chain": np.arange(n_chains),
            "draw": np.arange(n_samples),
        }

        # Create posterior dataset
        posterior_dict = {}
        for var_name, samples in posterior_samples.items():
            posterior_dict[var_name] = (["chain", "draw"], samples)

        posterior = xr.Dataset(posterior_dict, coords=coords)

        # Create sample_stats dataset if we have statistics
        sample_stats_ds = None
        if sample_stats:
            sample_stats_dict = {}
            for stat_name, values in sample_stats.items():
                sample_stats_dict[stat_name] = (["chain", "draw"], values)
            sample_stats_ds = xr.Dataset(sample_stats_dict, coords=coords)

        # Add sampling metadata as attributes
        posterior.attrs["sampling_method"] = "Gibbs Sampling"
        posterior.attrs["burn_in"] = burn_in
        posterior.attrs["thin"] = thin

        # Create InferenceData
        idata = az.InferenceData(
            posterior=posterior,
            sample_stats=sample_stats_ds if sample_stats_ds is not None else None,
        )

        return idata
