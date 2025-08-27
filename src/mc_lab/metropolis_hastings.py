"""
Metropolis-Hastings MCMC sampler with Random Walk proposal.

This module implements the Metropolis-Hastings algorithm for sampling from
probability distributions using a symmetric Random Walk proposal. The implementation
prioritizes educational clarity and includes full ArviZ integration for comprehensive
MCMC diagnostics.

The sampler uses log-probability functions for numerical stability and supports
adaptive proposal scaling for optimal acceptance rates.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Dict, List, Optional, Union

import arviz as az
import numpy as np
import xarray as xr
from tqdm.auto import tqdm

from ._rng import as_generator

__all__ = ["MetropolisHastingsSampler"]


class MetropolisHastingsSampler:
    """
    Metropolis-Hastings sampler with Random Walk proposal and ArviZ integration.

    This sampler uses a symmetric random walk proposal where new states are proposed
    as x' = x + ε, where ε ~ N(0, σ²I). The acceptance probability simplifies to
    min(1, π(x')/π(x)) due to the symmetric proposal.

    The implementation includes adaptive proposal scaling during burn-in to achieve
    optimal acceptance rates and returns results as ArviZ InferenceData for
    comprehensive diagnostics.

    Parameters
    ----------
    log_target : callable
        Function that computes log π(x) for the target distribution.
        Should handle both scalar and array inputs consistently.
    proposal_scale : float or array-like, default=1.0
        Standard deviation of the random walk proposal. Can be scalar
        for isotropic proposals or array for dimension-specific scaling.
    var_names : list of str, optional
        Names for the sampled variables. Auto-generated if None.
    adaptive_scaling : bool, default=True
        Whether to adapt proposal scale during burn-in for optimal acceptance.
    target_acceptance_rate : float, default=0.35
        Target acceptance rate for adaptive scaling. 0.35 is near-optimal
        for many problems (Gelman et al. 1996).

    Examples
    --------
    Sample from a 1D normal distribution:

    >>> def log_normal(x):
    ...     return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)
    >>> sampler = MetropolisHastingsSampler(log_normal, proposal_scale=0.8)
    >>> idata = sampler.sample(1000, n_chains=2)
    >>> print(az.summary(idata))

    Sample from a 2D correlated normal:

    >>> def log_mvn(x):
    ...     mu = np.array([1.0, -0.5])
    ...     cov_inv = np.array([[1.2, -0.4], [-0.4, 0.8]])
    ...     diff = x - mu
    ...     return -0.5 * diff @ cov_inv @ diff
    >>> sampler = MetropolisHastingsSampler(
    ...     log_mvn, proposal_scale=[0.6, 0.8], var_names=['x', 'y']
    ... )
    >>> idata = sampler.sample(2000, burn_in=500)
    """

    def __init__(
        self,
        log_target: Callable[[np.ndarray], float],
        proposal_scale: Union[float, np.ndarray] = 1.0,
        var_names: Optional[List[str]] = None,
        adaptive_scaling: bool = True,
        target_acceptance_rate: float = 0.35,
    ):
        self.log_target = log_target
        self.proposal_scale = np.atleast_1d(proposal_scale)
        self.var_names = var_names
        self.adaptive_scaling = adaptive_scaling
        self.target_acceptance_rate = target_acceptance_rate

        # Adaptive scaling parameters
        self.adaptation_window = 50
        self.adaptation_rate = 0.1  # How aggressively to adapt
        self.min_scale = 1e-6
        self.max_scale = 100.0

        # Will be set during sampling
        self._n_dim = None
        self._current_scales = None

    def sample(
        self,
        n_samples: int = 1000,
        n_chains: int = 4,
        burn_in: int = 1000,
        thin: int = 1,
        initial_states: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
    ) -> az.InferenceData:
        """
        Generate samples using Metropolis-Hastings algorithm.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate per chain (after burn-in and thinning).
        n_chains : int, default=4
            Number of independent chains to run.
        burn_in : int, default=1000
            Number of initial samples to discard per chain.
        thin : int, default=1
            Keep every 'thin'-th sample to reduce autocorrelation.
        initial_states : array-like, optional
            Initial states for chains. Shape should be (n_chains, n_dim) or (n_dim,).
            If (n_dim,), the same initial state is used for all chains with small
            random perturbation. If None, generates random initial states.
        random_seed : int, optional
            Random seed for reproducibility.
        progressbar : bool, default=True
            Show progress bar during sampling.

        Returns
        -------
        idata : arviz.InferenceData
            InferenceData object containing posterior samples and diagnostics.

        Notes
        -----
        The algorithm proceeds as follows for each chain:

        1. Initialize chain at starting state
        2. For each iteration:
           a. Propose new state: x' = x + N(0, σ²I)
           b. Compute acceptance probability: α = min(1, exp(log π(x') - log π(x)))
           c. Accept/reject with probability α
           d. Adapt proposal scale during burn-in (if enabled)
           e. Store sample if past burn-in and at thinning interval
        """
        rng = as_generator(random_seed)

        # Initialize chains
        initial_states = self._setup_initial_states(initial_states, n_chains, rng)
        self._n_dim = initial_states.shape[1]

        # Setup proposal scaling
        if len(self.proposal_scale) == 1:
            self._current_scales = np.full(
                (n_chains, self._n_dim), self.proposal_scale[0]
            )
        else:
            if len(self.proposal_scale) != self._n_dim:
                raise ValueError(
                    f"proposal_scale length {len(self.proposal_scale)} "
                    f"doesn't match dimension {self._n_dim}"
                )
            self._current_scales = np.tile(self.proposal_scale, (n_chains, 1))

        # Setup variable names
        if self.var_names is None:
            if self._n_dim == 1:
                self.var_names = ["x"]
            else:
                self.var_names = [f"x{i}" for i in range(self._n_dim)]
        elif len(self.var_names) != self._n_dim:
            raise ValueError(
                f"var_names length {len(self.var_names)} "
                f"doesn't match dimension {self._n_dim}"
            )

        # Storage
        total_iterations = burn_in + n_samples * thin
        posterior_samples = {
            name: np.zeros((n_chains, n_samples)) for name in self.var_names
        }

        sample_stats = {
            "log_likelihood": np.zeros((n_chains, n_samples)),
            "accepted": np.zeros((n_chains, n_samples), dtype=bool),
            "proposal_scale": np.zeros((n_chains, n_samples, self._n_dim)),
        }

        # Run chains
        for chain_idx in range(n_chains):
            self._run_chain(
                chain_idx,
                initial_states[chain_idx],
                n_samples,
                burn_in,
                thin,
                total_iterations,
                posterior_samples,
                sample_stats,
                rng,
                progressbar,
            )

        # Create InferenceData
        return self._create_inference_data(
            posterior_samples,
            sample_stats,
            n_chains,
            n_samples,
            burn_in,
            thin,
        )

    def _setup_initial_states(
        self,
        initial_states: Optional[np.ndarray],
        n_chains: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Setup initial states for all chains."""
        if initial_states is None:
            # Test dimension with dummy evaluation
            test_state = rng.standard_normal(1)
            try:
                self.log_target(test_state)
                n_dim = 1
            except (ValueError, TypeError):
                # Try higher dimensions
                for dim in [2, 3, 4, 5]:
                    try:
                        test_state = rng.standard_normal(dim)
                        self.log_target(test_state)
                        n_dim = dim
                        break
                    except (ValueError, TypeError):
                        continue
                else:
                    raise ValueError("Cannot determine target function dimensionality")

            # Generate overdispersed initial states
            initial_states = rng.standard_normal((n_chains, n_dim)) * 2.0

        else:
            initial_states = np.atleast_2d(initial_states)
            if initial_states.shape[0] == 1 and n_chains > 1:
                # Replicate and add small perturbations
                perturbations = (
                    rng.standard_normal((n_chains, initial_states.shape[1])) * 0.1
                )
                initial_states = np.tile(initial_states, (n_chains, 1)) + perturbations

        return initial_states

    def _run_chain(
        self,
        chain_idx: int,
        initial_state: np.ndarray,
        n_samples: int,
        burn_in: int,
        thin: int,
        total_iterations: int,
        posterior_samples: Dict[str, np.ndarray],
        sample_stats: Dict[str, np.ndarray],
        rng: np.random.Generator,
        progressbar: bool,
    ) -> None:
        """Run a single MCMC chain."""
        current_state = initial_state.copy()
        current_log_prob = self.log_target(current_state)

        # Adaptive scaling tracking
        recent_accepts = []
        sample_idx = 0

        # Progress bar
        pbar = None
        if progressbar:
            pbar = tqdm(
                total=total_iterations,
                desc=f"Chain {chain_idx + 1}",
                leave=True,
                dynamic_ncols=True,
            )

        for iteration in range(total_iterations):
            # Propose new state
            proposal = current_state + rng.normal(0, self._current_scales[chain_idx])

            try:
                proposal_log_prob = self.log_target(proposal)

                # Compute acceptance probability
                log_alpha = proposal_log_prob - current_log_prob
                accept = (log_alpha >= 0) or (rng.random() < np.exp(log_alpha))

            except (ValueError, OverflowError, np.linalg.LinAlgError):
                # Numerical issues - reject proposal
                accept = False
                proposal_log_prob = -np.inf

            # Update state
            if accept:
                current_state = proposal
                current_log_prob = proposal_log_prob

            recent_accepts.append(accept)

            # Adaptive scaling during burn-in
            if (
                self.adaptive_scaling
                and iteration < burn_in
                and len(recent_accepts) >= self.adaptation_window
            ):
                acceptance_rate = np.mean(recent_accepts)
                self._adapt_proposal_scale(chain_idx, acceptance_rate)
                recent_accepts = []  # Reset window

            # Store sample if past burn-in and at thinning interval
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                for dim_idx, var_name in enumerate(self.var_names):
                    if self._n_dim == 1:
                        posterior_samples[var_name][chain_idx, sample_idx] = (
                            current_state[0]
                        )
                    else:
                        posterior_samples[var_name][chain_idx, sample_idx] = (
                            current_state[dim_idx]
                        )

                sample_stats["log_likelihood"][chain_idx, sample_idx] = current_log_prob
                sample_stats["accepted"][chain_idx, sample_idx] = accept
                sample_stats["proposal_scale"][chain_idx, sample_idx] = (
                    self._current_scales[chain_idx]
                )

                sample_idx += 1

                if pbar is not None:
                    current_accept_rate = np.mean(
                        sample_stats["accepted"][chain_idx, :sample_idx]
                    )
                    pbar.set_postfix(
                        samples=sample_idx,
                        accept_rate=f"{current_accept_rate:.3f}",
                    )

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    def _adapt_proposal_scale(self, chain_idx: int, acceptance_rate: float) -> None:
        """Adapt proposal scale based on recent acceptance rate."""
        target_rate = self.target_acceptance_rate

        if acceptance_rate > target_rate + 0.05:
            # Too many accepts - increase scale
            factor = 1 + self.adaptation_rate
        elif acceptance_rate < target_rate - 0.05:
            # Too few accepts - decrease scale
            factor = 1 - self.adaptation_rate
        else:
            # Acceptance rate is good
            return

        self._current_scales[chain_idx] *= factor

        # Clamp to reasonable bounds
        self._current_scales[chain_idx] = np.clip(
            self._current_scales[chain_idx],
            self.min_scale,
            self.max_scale,
        )

    def _create_inference_data(
        self,
        posterior_samples: Dict[str, np.ndarray],
        sample_stats: Dict[str, np.ndarray],
        n_chains: int,
        n_samples: int,
        burn_in: int,
        thin: int,
    ) -> az.InferenceData:
        """Create ArviZ InferenceData object from samples."""
        # Create coordinates
        coords = {
            "chain": np.arange(n_chains),
            "draw": np.arange(n_samples),
        }

        if self._n_dim > 1:
            coords["dim"] = np.arange(self._n_dim)

        # Create posterior dataset
        posterior_dict = {}
        for var_name, samples in posterior_samples.items():
            posterior_dict[var_name] = (["chain", "draw"], samples)

        posterior = xr.Dataset(posterior_dict, coords=coords)

        # Create sample_stats dataset
        sample_stats_dict = {}
        for stat_name, values in sample_stats.items():
            if stat_name == "proposal_scale":
                if self._n_dim == 1:
                    sample_stats_dict[stat_name] = (["chain", "draw"], values[:, :, 0])
                else:
                    sample_stats_dict[stat_name] = (["chain", "draw", "dim"], values)
            else:
                sample_stats_dict[stat_name] = (["chain", "draw"], values)

        sample_stats_ds = xr.Dataset(sample_stats_dict, coords=coords)

        # Add metadata
        posterior.attrs.update(
            {
                "sampling_method": "Metropolis-Hastings",
                "proposal_type": "Random Walk",
                "burn_in": burn_in,
                "thin": thin,
                "adaptive_scaling": self.adaptive_scaling,
                "target_acceptance_rate": self.target_acceptance_rate,
            }
        )

        # Create InferenceData
        idata = az.InferenceData(
            posterior=posterior,
            sample_stats=sample_stats_ds,
        )

        return idata

    def get_acceptance_rates(self, idata: az.InferenceData) -> Dict[str, float]:
        """
        Compute acceptance rates from InferenceData.

        Parameters
        ----------
        idata : arviz.InferenceData
            InferenceData from sampling.

        Returns
        -------
        rates : dict
            Acceptance rates per chain and overall.
        """
        accepted = idata.sample_stats["accepted"].values

        rates = {}
        for chain in range(accepted.shape[0]):
            rates[f"chain_{chain}"] = float(np.mean(accepted[chain]))

        rates["overall"] = float(np.mean(accepted))

        return rates

    def tune_proposal_scale(
        self,
        initial_scale: Union[float, np.ndarray] = 1.0,
        target_samples: int = 1000,
        random_seed: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """
        Tune proposal scale to achieve target acceptance rate.

        This is a standalone tuning method that can be run before main sampling
        to find good proposal scales.

        Parameters
        ----------
        initial_scale : float or array-like, default=1.0
            Starting proposal scale.
        target_samples : int, default=1000
            Number of samples to use for tuning.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        tuned_scale : float or array
            Tuned proposal scale(s).
        """
        # Temporarily set parameters for tuning
        original_scale = self.proposal_scale
        original_adaptive = self.adaptive_scaling

        self.proposal_scale = np.atleast_1d(initial_scale)
        self.adaptive_scaling = True

        try:
            # Run short sampling for tuning
            idata = self.sample(
                n_samples=target_samples,
                n_chains=1,
                burn_in=target_samples // 2,
                thin=1,
                random_seed=random_seed,
                progressbar=False,
            )

            # Extract final proposal scale
            final_scale = idata.sample_stats["proposal_scale"].values[0, -1]

            if len(final_scale) == 1:
                return float(final_scale[0])
            else:
                return final_scale

        finally:
            # Restore original parameters
            self.proposal_scale = original_scale
            self.adaptive_scaling = original_adaptive
