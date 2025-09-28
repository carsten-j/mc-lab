"""
Barker's MCMC sampler with Gaussian proposal.

This module implements Barker's MCMC algorithm for sampling from probability
distributions using a symmetric Gaussian proposal. The Barker algorithm uses
a different acceptance rule than Metropolis-Hastings, where the acceptance
probability is π(X*) / (π(X*) + π(Xₙ)).

The implementation prioritizes educational clarity and includes full ArviZ
integration for comprehensive MCMC diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Dict, List, Optional, Union

import arviz as az
import numpy as np
from tqdm.auto import tqdm

from ._inference_data import create_inference_data
from ._rng import as_generator

__all__ = ["BarkerMCMCSampler"]


class BarkerMCMCSampler:
    """
    Barker's MCMC sampler with Gaussian proposal and ArviZ integration.

    This sampler uses Barker's acceptance rule with a symmetric Gaussian proposal
    where new states are proposed as x' = x + ε, where ε ~ N(0, σ²I).
    The acceptance probability is π(x') / (π(x') + π(x)) instead of the
    Metropolis-Hastings min(1, π(x')/π(x)).

    The Barker algorithm often provides better mixing properties compared to
    standard Metropolis-Hastings, especially for multi-modal distributions.

    Parameters
    ----------
    log_target : callable
        Function that computes log π(x) for the target distribution.
        Should handle both scalar and array inputs consistently.
    proposal_scale : float or array-like, default=1.0
        Standard deviation of the Gaussian proposal. Can be scalar
        for isotropic proposals or array for dimension-specific scaling.
    var_names : list of str, optional
        Names for the sampled variables. Auto-generated if None.
    adaptive_scaling : bool, default=True
        Whether to adapt proposal scale during burn-in for optimal acceptance.
    target_acceptance_rate : float, default=0.5
        Target acceptance rate for adaptive scaling. For Barker MCMC,
        0.5 is often optimal due to the different acceptance rule.

    Examples
    --------
    Sample from a 1D normal distribution:

    >>> def log_normal(x):
    ...     return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)
    >>> sampler = BarkerMCMCSampler(log_normal, proposal_scale=0.8)
    >>> idata = sampler.sample(1000, n_chains=2)
    >>> print(az.summary(idata))

    Sample from a 2D correlated normal:

    >>> def log_mvn(x):
    ...     mu = np.array([1.0, -0.5])
    ...     cov_inv = np.array([[1.2, -0.4], [-0.4, 0.8]])
    ...     diff = x - mu
    ...     return -0.5 * diff @ cov_inv @ diff
    >>> sampler = BarkerMCMCSampler(
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
        target_acceptance_rate: float = 0.5,
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
        Generate samples using Barker's MCMC algorithm.

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
        The Barker algorithm proceeds as follows for each chain:

        1. Initialize chain at starting state
        2. For each iteration:
           a. Propose new state: x' = x + N(0, σ²I)
           b. Compute acceptance probability: α = π(x') / (π(x') + π(x))
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
            "acceptance_prob": np.zeros(
                (n_chains, n_samples)
            ),  # Store Barker acceptance prob
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
            # Determine dimensionality from var_names or proposal_scale if available
            if self.var_names is not None:
                n_dim = len(self.var_names)
            elif len(self.proposal_scale) > 1:
                n_dim = len(self.proposal_scale)
            else:
                # Test dimension with dummy evaluation
                test_state = rng.standard_normal(1)
                try:
                    result = self.log_target(test_state)
                    # Check if result is a single value (scalar or 1-element array)
                    result = np.asarray(result)
                    if result.size == 1 and np.isfinite(result):
                        n_dim = 1
                    else:
                        raise ValueError("Target function doesn't return single value")
                except (ValueError, TypeError, IndexError):
                    # Try higher dimensions
                    for dim in [2, 3, 4, 5]:
                        try:
                            test_state = rng.standard_normal(dim)
                            result = self.log_target(test_state)
                            result = np.asarray(result)
                            if result.size == 1 and np.isfinite(result):
                                n_dim = dim
                                break
                        except (ValueError, TypeError, IndexError):
                            continue
                    else:
                        raise ValueError(
                            "Cannot determine target function dimensionality"
                        )

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
        """Run a single MCMC chain using Barker's algorithm."""
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
                desc=f"Barker Chain {chain_idx + 1}",
                leave=True,
                dynamic_ncols=True,
                disable=False,
                file=None,  # Let tqdm choose the best output stream
                ascii=False,
            )

        for iteration in range(total_iterations):
            # Propose new state using Gaussian proposal
            proposal = current_state + rng.normal(0, self._current_scales[chain_idx])

            try:
                proposal_log_prob = self.log_target(proposal)

                # Barker's acceptance probability: π(X*) / (π(X*) + π(Xₙ))
                # In log space: exp(log π(X*)) / (exp(log π(X*)) + exp(log π(Xₙ)))
                # Using log-sum-exp trick for numerical stability
                log_sum = np.logaddexp(proposal_log_prob, current_log_prob)
                barker_acceptance_prob = np.exp(proposal_log_prob - log_sum)

                # Accept with Barker probability
                accept = rng.random() < barker_acceptance_prob

            except (ValueError, OverflowError, np.linalg.LinAlgError):
                # Numerical issues - reject proposal
                accept = False
                proposal_log_prob = -np.inf
                barker_acceptance_prob = 0.0

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

                sample_stats["log_likelihood"][chain_idx, sample_idx] = (
                    float(current_log_prob)
                    if np.ndim(current_log_prob) == 0
                    else float(current_log_prob[0])
                )
                sample_stats["accepted"][chain_idx, sample_idx] = accept
                sample_stats["proposal_scale"][chain_idx, sample_idx] = (
                    self._current_scales[chain_idx].copy()
                )
                # Store Barker-specific acceptance probability
                try:
                    sample_stats["acceptance_prob"][chain_idx, sample_idx] = float(
                        barker_acceptance_prob
                    )
                except UnboundLocalError:
                    sample_stats["acceptance_prob"][chain_idx, sample_idx] = 0.0

                sample_idx += 1

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                # Update postfix with current statistics if we have samples
                if sample_idx > 0:
                    current_accept_rate = np.mean(
                        sample_stats["accepted"][chain_idx, :sample_idx]
                    )
                    pbar.set_postfix(
                        samples=sample_idx,
                        accept_rate=f"{current_accept_rate:.3f}",
                    )

        if pbar is not None:
            pbar.close()
            # Small delay to prevent display race conditions in notebooks
            try:
                import time

                time.sleep(0.01)  # 10ms delay
            except ImportError:
                pass

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
        return create_inference_data(
            posterior_samples=posterior_samples,
            sample_stats=sample_stats,
            n_chains=n_chains,
            n_samples=n_samples,
            n_dim=self._n_dim,
            algorithm_name="Barker MCMC",
            burn_in=burn_in,
            thin=thin,
            proposal_type="Gaussian",
            adaptive_scaling=self.adaptive_scaling,
            target_acceptance_rate=self.target_acceptance_rate,
        )

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

    def get_barker_acceptance_probs(self, idata: az.InferenceData) -> Dict[str, float]:
        """
        Compute mean Barker acceptance probabilities from InferenceData.

        Parameters
        ----------
        idata : arviz.InferenceData
            InferenceData from sampling.

        Returns
        -------
        probs : dict
            Mean acceptance probabilities per chain and overall.
        """
        acceptance_probs = idata.sample_stats["acceptance_prob"].values

        probs = {}
        for chain in range(acceptance_probs.shape[0]):
            probs[f"chain_{chain}"] = float(np.mean(acceptance_probs[chain]))

        probs["overall"] = float(np.mean(acceptance_probs))

        return probs
