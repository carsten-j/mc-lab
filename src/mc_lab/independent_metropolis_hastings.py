"""
Independent Metropolis-Hastings MCMC sampler.

This module implements the independent Metropolis-Hastings algorithm for sampling from
probability distributions using independent proposals. Unlike the random walk version,
the proposal distribution is independent of the current state, allowing for more
efficient sampling when good proposal distributions are available.

The implementation prioritizes educational clarity and includes full ArviZ integration
for comprehensive MCMC diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Dict, List, Optional, Union

import arviz as az
import numpy as np
from tqdm.auto import tqdm

from ._inference_data import create_inference_data
from ._rng import as_generator

__all__ = ["IndependentMetropolisHastingsSampler", "independent_metropolis_hastings"]


class IndependentMetropolisHastingsSampler:
    """
    Independent Metropolis-Hastings sampler with ArviZ integration.

    This sampler implements the independent Metropolis-Hastings algorithm where
    proposals are drawn independently of the current state. The acceptance probability
    is given by:

    α(x, x') = min(1, [π(x') / π(x)] × [q(x) / q(x')])

    where π is the target distribution and q is the proposal distribution.

    This approach can be very efficient when a good approximation to the target
    distribution is available as the proposal distribution. Unlike random walk
    Metropolis-Hastings, the proposal is independent of the current state.

    The implementation includes comprehensive diagnostics through ArviZ integration
    and supports multiple chains for convergence assessment.

    Parameters
    ----------
    target_log_pdf : callable
        Function that computes log π(x) for the target distribution.
        Should handle both scalar and array inputs consistently.
        Must accept a single argument (the state) and return a float.
    proposal_sampler : callable
        Function that generates samples from the proposal distribution q(x).
        Should return a single sample when called with no arguments.
        For reproducibility, should accept an optional random state parameter.
    proposal_log_pdf : callable
        Function that computes log q(x) for the proposal distribution.
        Should handle both scalar and array inputs consistently.
        Must accept a single argument (the state) and return a float.
    var_names : list of str, optional
        Names for the sampled variables. Auto-generated if None.
        For multidimensional problems, should have length equal to the dimension.

    Attributes
    ----------
    target_log_pdf : callable
        The target log probability density function.
    proposal_sampler : callable
        The proposal sampling function.
    proposal_log_pdf : callable
        The proposal log probability density function.
    var_names : list of str
        Variable names for the sampled parameters.

    Examples
    --------
    Sample from a standard normal using a normal proposal:

    >>> import numpy as np
    >>> from scipy import stats
    >>>
    >>> def target_log_pdf(x):
    ...     return stats.norm.logpdf(x, loc=0, scale=1)
    >>>
    >>> def proposal_sampler():
    ...     return np.random.normal(0, 1.2)
    >>>
    >>> def proposal_log_pdf(x):
    ...     return stats.norm.logpdf(x, loc=0, scale=1.2)
    >>>
    >>> sampler = IndependentMetropolisHastingsSampler(
    ...     target_log_pdf, proposal_sampler, proposal_log_pdf
    ... )
    >>> idata = sampler.sample(n_samples=1000, n_chains=4)
    >>> print(az.summary(idata))

    Sample from a shifted distribution using importance sampling:

    >>> def shifted_target_log_pdf(x):
    ...     return stats.norm.logpdf(x, loc=2.0, scale=1.0)
    >>>
    >>> def centered_proposal_sampler():
    ...     return np.random.normal(0, 1.5)
    >>>
    >>> def centered_proposal_log_pdf(x):
    ...     return stats.norm.logpdf(x, loc=0, scale=1.5)
    >>>
    >>> sampler = IndependentMetropolisHastingsSampler(
    ...     shifted_target_log_pdf,
    ...     centered_proposal_sampler,
    ...     centered_proposal_log_pdf,
    ...     var_names=['shifted_x']
    ... )
    >>> idata = sampler.sample(n_samples=2000, burn_in=500)

    Notes
    -----
    The independent Metropolis-Hastings algorithm differs from the random walk
    version in that proposals are generated independently of the current state.
    This can lead to:

    1. **Better mixing** when the proposal distribution is a good approximation
       to the target distribution.
    2. **Poor performance** when the proposal distribution has lighter tails
       than the target distribution.
    3. **More efficient exploration** of multimodal distributions if the proposal
       covers all modes adequately.

    **Choice of Proposal Distribution:**

    - The proposal should have heavier tails than the target for good performance
    - A good rule of thumb is to use a proposal with 1.5-2x the scale of the target
    - For multimodal targets, mixtures of distributions can be effective proposals

    **Convergence Diagnostics:**

    The returned InferenceData object includes standard MCMC diagnostics:

    - R-hat (potential scale reduction factor)
    - Effective sample size (ESS)
    - Monte Carlo standard error
    - Trace plots and other visualizations via ArviZ

    References
    ----------
    .. [1] Metropolis, N., et al. (1953). "Equation of state calculations by fast
           computing machines." The journal of chemical physics, 21(6), 1087-1092.
    .. [2] Hastings, W. K. (1970). "Monte Carlo sampling methods using Markov
           chains and their applications." Biometrika, 57(1), 97-109.
    .. [3] Robert, C., & Casella, G. (2013). "Monte Carlo statistical methods."
           Springer Science & Business Media.
    """

    def __init__(
        self,
        target_log_pdf: Callable[[Union[float, np.ndarray]], float],
        proposal_sampler: Callable[[], Union[float, np.ndarray]],
        proposal_log_pdf: Callable[[Union[float, np.ndarray]], float],
        var_names: Optional[List[str]] = None,
    ):
        self.target_log_pdf = target_log_pdf
        self.proposal_sampler = proposal_sampler
        self.proposal_log_pdf = proposal_log_pdf
        self.var_names = var_names

        # Will be set during sampling
        self._n_dim = None

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
        Generate samples using Independent Metropolis-Hastings algorithm.

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
            If (n_dim,), the same initial state is used for all chains.
            If None, generates initial states using the proposal sampler.
        random_seed : int, optional
            Random seed for reproducibility.
        progressbar : bool, default=True
            Show progress bar during sampling.

        Returns
        -------
        idata : arviz.InferenceData
            InferenceData object containing posterior samples and diagnostics.
            Includes acceptance rates, log-likelihood values, and other statistics.

        Notes
        -----
        The algorithm proceeds as follows for each chain:

        1. Initialize chain at starting state
        2. For each iteration:
           a. Propose new state: x' ~ q(x') (independent of current state)
           b. Compute acceptance probability: α = min(1, [π(x')/π(x)] × [q(x)/q(x')])
           c. Accept/reject with probability α
           d. Store sample if past burn-in and at thinning interval

        **Performance Tips:**

        - Use burn_in ≥ 1000 for complex distributions
        - Monitor acceptance rates: 20-50% is typically good for independent MH
        - Higher acceptance rates may indicate the proposal is too similar to target
        - Lower acceptance rates may indicate poor proposal choice

        **Diagnostics:**

        Check the returned InferenceData for:

        - R-hat < 1.1 for all parameters (indicates convergence)
        - ESS > 100 per chain (indicates adequate mixing)
        - Trace plots should show good mixing across chains
        """
        rng = as_generator(random_seed)

        # Generate initial sample to determine dimensionality
        test_sample = self.proposal_sampler()
        if np.isscalar(test_sample):
            self._n_dim = 1
            test_sample = float(test_sample)
        else:
            test_sample = np.atleast_1d(test_sample)
            self._n_dim = len(test_sample)

        # Setup initial states
        initial_states = self._setup_initial_states(initial_states, n_chains, rng)

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
            "proposal_log_pdf": np.zeros((n_chains, n_samples)),
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
            # Generate initial states using proposal sampler
            states = []
            for _ in range(n_chains):
                sample = self.proposal_sampler()
                if self._n_dim == 1:
                    states.append([float(sample)])
                else:
                    states.append(np.atleast_1d(sample))
            return np.array(states)

        initial_states = np.atleast_2d(initial_states)
        if initial_states.shape[0] == 1 and n_chains > 1:
            # Broadcast single initial state to all chains
            return np.tile(initial_states, (n_chains, 1))
        elif initial_states.shape[0] != n_chains:
            raise ValueError(
                f"initial_states shape {initial_states.shape} doesn't match "
                f"n_chains {n_chains}"
            )

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
    ):
        """Run a single MCMC chain."""
        current_state = initial_state.copy()
        if self._n_dim == 1:
            current_state = current_state[0]

        current_log_target = self.target_log_pdf(current_state)
        current_log_proposal = self.proposal_log_pdf(current_state)

        sample_idx = 0

        # Setup progress bar
        pbar = None
        if progressbar:
            n_total_chains = posterior_samples[self.var_names[0]].shape[0]
            pbar = tqdm(
                total=total_iterations,
                desc=f"Chain {chain_idx + 1}/{n_total_chains}",
                unit="samples",
                leave=True,
            )

        for iteration in range(total_iterations):
            # Generate proposal (independent of current state)
            proposed_state = self.proposal_sampler()
            if self._n_dim == 1:
                proposed_state = float(proposed_state)
            else:
                proposed_state = np.atleast_1d(proposed_state)

            # Calculate log acceptance ratio
            try:
                proposed_log_target = self.target_log_pdf(proposed_state)
                proposed_log_proposal = self.proposal_log_pdf(proposed_state)

                log_ratio = (
                    proposed_log_target
                    - current_log_target
                    + current_log_proposal
                    - proposed_log_proposal
                )

                # Accept or reject
                if np.log(rng.random()) < log_ratio:
                    current_state = proposed_state
                    current_log_target = proposed_log_target
                    current_log_proposal = proposed_log_proposal
                    accepted = True
                else:
                    accepted = False

            except (ValueError, OverflowError, ZeroDivisionError):
                # Reject proposals that cause numerical issues
                accepted = False
                proposed_log_target = -np.inf
                proposed_log_proposal = current_log_proposal

            # Store sample if past burn-in and at thinning interval
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                if self._n_dim == 1:
                    posterior_samples[self.var_names[0]][chain_idx, sample_idx] = (
                        current_state
                    )
                else:
                    for i, name in enumerate(self.var_names):
                        posterior_samples[name][chain_idx, sample_idx] = current_state[
                            i
                        ]

                sample_stats["log_likelihood"][chain_idx, sample_idx] = (
                    current_log_target
                )
                sample_stats["accepted"][chain_idx, sample_idx] = accepted
                sample_stats["proposal_log_pdf"][chain_idx, sample_idx] = (
                    current_log_proposal
                )

                sample_idx += 1

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

    def _create_inference_data(
        self,
        posterior_samples: Dict[str, np.ndarray],
        sample_stats: Dict[str, np.ndarray],
        n_chains: int,
        n_samples: int,
        burn_in: int,
        thin: int,
    ) -> az.InferenceData:
        """Create ArviZ InferenceData object."""
        return create_inference_data(
            posterior_samples=posterior_samples,
            sample_stats=sample_stats,
            n_chains=n_chains,
            n_samples=n_samples,
            n_dim=self._n_dim,
            algorithm_name="Independent Metropolis-Hastings",
            burn_in=burn_in,
            thin=thin,
        )

    def get_acceptance_rates(self, idata: az.InferenceData) -> Dict[str, float]:
        """
        Calculate acceptance rates from InferenceData.

        Parameters
        ----------
        idata : arviz.InferenceData
            InferenceData object from sampling.

        Returns
        -------
        dict
            Dictionary with 'overall' acceptance rate and per-chain rates.
        """
        accepted = idata.sample_stats["accepted"].values

        rates = {"overall": float(np.mean(accepted))}

        for chain_idx in range(accepted.shape[0]):
            rates[f"chain_{chain_idx}"] = float(np.mean(accepted[chain_idx]))

        return rates


def independent_metropolis_hastings(
    target_log_pdf: Callable[[Union[float, np.ndarray]], float],
    proposal_sampler: Callable[[], Union[float, np.ndarray]],
    proposal_log_pdf: Callable[[Union[float, np.ndarray]], float],
    n_samples: int,
    initial_value: Union[float, np.ndarray] = 0.0,
) -> tuple[np.ndarray, int]:
    """
    Independent Metropolis-Hastings sampler (functional interface).

    This is a simple functional implementation of the independent Metropolis-Hastings
    algorithm. For more advanced features like multiple chains, ArviZ integration,
    and comprehensive diagnostics, use the IndependentMetropolisHastingsSampler class.

    Parameters
    ----------
    target_log_pdf : callable
        Function that computes log π(x) for the target distribution.
        Should accept a single argument (the state) and return a float.
    proposal_sampler : callable
        Function that generates samples from the proposal distribution q(x).
        Should return a single sample when called with no arguments.
    proposal_log_pdf : callable
        Function that computes log q(x) for the proposal distribution.
        Should accept a single argument (the state) and return a float.
    n_samples : int
        Number of samples to generate from the target distribution.
    initial_value : float or array-like, default=0.0
        Starting value for the Markov chain. Should be in the support
        of the target distribution.

    Returns
    -------
    samples : ndarray
        Array of shape (n_samples,) containing samples from the target distribution.
        Each element represents one sample from the Markov chain.
    accepted : int
        Number of accepted proposals out of n_samples total proposals.
        Acceptance rate can be computed as accepted / n_samples.

    Notes
    -----
    The independent Metropolis-Hastings algorithm generates proposals independently
    of the current state, using the acceptance probability:

    α(x, x') = min(1, [π(x') / π(x)] × [q(x) / q(x')])

    where π is the target distribution and q is the proposal distribution.

    **Choice of Proposal Distribution:**

    The proposal distribution should ideally:
    - Have heavier tails than the target distribution
    - Have reasonable overlap with the target distribution
    - Be easy to sample from and evaluate

    **Performance Considerations:**

    - Acceptance rates of 20-50% are typically good for independent MH
    - Very high acceptance rates may indicate inefficient exploration
    - Very low acceptance rates may indicate poor proposal choice

    Examples
    --------
    Sample from a standard normal distribution:

    >>> import numpy as np
    >>> from scipy import stats
    >>>
    >>> def target_log_pdf(x):
    ...     return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)
    >>>
    >>> def proposal_sampler():
    ...     return np.random.normal(0, 1.2)
    >>>
    >>> def proposal_log_pdf(x):
    ...     return stats.norm.logpdf(x, loc=0, scale=1.2)
    >>>
    >>> samples, n_accepted = independent_metropolis_hastings(
    ...     target_log_pdf, proposal_sampler, proposal_log_pdf, 1000
    ... )
    >>> print(f"Acceptance rate: {n_accepted / 1000:.2f}")
    >>> print(f"Sample mean: {np.mean(samples):.3f}")
    >>> print(f"Sample std: {np.std(samples):.3f}")

    See Also
    --------
    IndependentMetropolisHastingsSampler : Class-based implementation with more features
    """
    samples = np.zeros(n_samples)
    current = initial_value
    accepted = 0

    for i in range(n_samples):
        # Generate proposal (independent of current state)
        proposed = proposal_sampler()

        # Calculate acceptance ratio (in log space for numerical stability)
        log_ratio = (
            target_log_pdf(proposed)
            - target_log_pdf(current)
            + proposal_log_pdf(current)
            - proposal_log_pdf(proposed)
        )

        # Accept or reject
        if np.log(np.random.rand()) < log_ratio:
            current = proposed
            accepted += 1

        samples[i] = current

    return samples, accepted
