"""Metropolis-adjusted Langevin Algorithm (MALA) MCMC sampler.

This module implements the MALA algorithm which uses gradient information to propose
new states, making it more efficient than random walk Metropolis-Hastings for smooth
target distributions where gradients are available.

The implementation prioritizes educational clarity and includes full ArviZ integration
for comprehensive MCMC diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Dict, List, Optional, Union

import arviz as az
import numpy as np
from scipy.stats import multivariate_normal
from tqdm.auto import tqdm

from ._rng import as_generator
from ._inference_data import create_inference_data

__all__ = ["MALASampler", "mala"]


class MALASampler:
    """
    Metropolis-adjusted Langevin Algorithm (MALA) sampler with ArviZ integration.

    MALA uses gradient information from the target distribution to propose new states,
    making it more efficient than random walk Metropolis-Hastings for smooth target
    distributions where gradients are available or can be computed.

    The algorithm uses Langevin dynamics to propose new states:
    x' = x + (ε²/2) * ∇log π(x) + ε * Z

    where ε is the step size, π is the target distribution, and Z ~ N(0, I).
    This gradient-guided proposal allows for more efficient exploration of the
    target distribution compared to random walk methods.

    The implementation includes comprehensive diagnostics through ArviZ integration
    and supports multiple chains for convergence assessment.

    Parameters
    ----------
    log_target : callable
        Function that computes log π(x) for the target distribution.
        Should handle both scalar and array inputs consistently.
        Must accept a single argument (the state) and return a float.
    grad_log_target : callable
        Function that computes the gradient ∇log π(x) of the log target density.
        Should return an array of the same shape as the input.
        Must accept a single argument (the state) and return an array.
    step_size : float, default=0.1
        Step size parameter ε for the Langevin dynamics.
        Controls the scale of the proposals and acceptance rate.
        Smaller values lead to higher acceptance but slower mixing.
    var_names : list of str, optional
        Names for the sampled variables. Auto-generated if None.
        For multidimensional problems, should have length equal to the dimension.

    Attributes
    ----------
    log_target : callable
        The target log probability density function.
    grad_log_target : callable
        The gradient of the target log probability density function.
    step_size : float
        The step size parameter for Langevin dynamics.
    var_names : list of str
        Variable names for the sampled parameters.

    Examples
    --------
    Sample from a 2D multivariate normal distribution:

    >>> import numpy as np
    >>> from scipy import stats
    >>>
    >>> # Define target distribution (2D normal with correlation)
    >>> mean = np.array([0.0, 0.0])
    >>> cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> cov_inv = np.linalg.inv(cov)
    >>>
    >>> def log_target(x):
    ...     diff = x - mean
    ...     return -0.5 * diff @ cov_inv @ diff
    >>>
    >>> def grad_log_target(x):
    ...     return -cov_inv @ (x - mean)
    >>>
    >>> sampler = MALASampler(
    ...     log_target=log_target,
    ...     grad_log_target=grad_log_target,
    ...     step_size=0.2,
    ...     var_names=['x', 'y']
    ... )
    >>> idata = sampler.sample(n_samples=1000, n_chains=4)
    >>> print(az.summary(idata))

    Sample from a 1D distribution with known gradient:

    >>> def log_normal(x):
    ...     return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)
    >>>
    >>> def grad_log_normal(x):
    ...     return -x
    >>>
    >>> sampler = MALASampler(
    ...     log_target=log_normal,
    ...     grad_log_target=grad_log_normal,
    ...     step_size=0.5
    ... )
    >>> idata = sampler.sample(n_samples=2000, burn_in=500)

    Notes
    -----
    **Algorithm Overview:**

    MALA combines the efficiency of gradient-based methods with the theoretical
    guarantees of MCMC:

    1. **Proposal Generation**: Use Langevin dynamics to propose new states
       x' = x + (ε²/2) * ∇log π(x) + ε * Z, where Z ~ N(0, I)
    2. **Acceptance Step**: Accept/reject using Metropolis criterion with
       asymmetric proposal correction
    3. **Gradient Information**: Exploits local structure of the target distribution

    **Advantages over Random Walk MH:**

    - More efficient exploration for smooth distributions
    - Better performance in high dimensions
    - Faster convergence when gradients are informative
    - Can achieve higher effective sample sizes

    **Tuning Guidelines:**

    - **Step size**: Controls trade-off between acceptance rate and step size
    - Target acceptance rate: 50-70% (higher than random walk MH)
    - Too small ε: High acceptance but inefficient exploration
    - Too large ε: Low acceptance due to overshooting

    **When to Use MALA:**

    - Target distribution is smooth and differentiable
    - Gradients are available or can be computed efficiently
    - Working with continuous distributions
    - Need efficient sampling in moderate to high dimensions

    **Convergence Diagnostics:**

    The returned InferenceData object includes standard MCMC diagnostics:

    - R-hat (potential scale reduction factor)
    - Effective sample size (ESS)
    - Monte Carlo standard error
    - Trace plots and other visualizations via ArviZ

    References
    ----------
    .. [1] Roberts, G. O., & Tweedie, R. L. (1996). "Exponential convergence of
           Langevin distributions and their discrete approximations."
           Bernoulli, 2(4), 341-363.
    .. [2] Roberts, G. O., & Rosenthal, J. S. (1998). "Optimal scaling of discrete
           approximations to Langevin diffusions." Journal of the Royal Statistical
           Society: Series B, 60(1), 255-268.
    .. [3] Xifara, T., et al. (2014). "Langevin diffusions and the Metropolis-adjusted
           Langevin algorithm." Statistics & Probability Letters, 91, 14-19.
    """

    def __init__(
        self,
        log_target: Callable[[Union[float, np.ndarray]], float],
        grad_log_target: Callable[[Union[float, np.ndarray]], np.ndarray],
        step_size: float = 0.1,
        var_names: Optional[List[str]] = None,
    ):
        self.log_target = log_target
        self.grad_log_target = grad_log_target
        self.step_size = step_size
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
        Generate samples using Metropolis-adjusted Langevin Algorithm.

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
            If None, generates initial states using standard normal.
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
        The MALA algorithm proceeds as follows for each chain:

        1. Initialize chain at starting state
        2. For each iteration:
           a. Propose new state using Langevin dynamics:
              x' = x + (ε²/2) * ∇log π(x) + ε * Z, where Z ~ N(0, I)
           b. Compute acceptance probability with proposal correction
           c. Accept/reject with computed probability
           d. Store sample if past burn-in and at thinning interval

        **Performance Tips:**

        - Use burn_in ≥ 1000 for complex distributions
        - Monitor acceptance rates: 50-70% is typically good for MALA
        - Tune step_size to achieve target acceptance rate
        - Smaller step sizes give higher acceptance but slower mixing

        **Diagnostics:**

        Check the returned InferenceData for:

        - R-hat < 1.1 for all parameters (indicates convergence)
        - ESS > 100 per chain (indicates adequate mixing)
        - Trace plots should show good mixing across chains
        """
        rng = as_generator(random_seed)

        # Generate initial sample to determine dimensionality
        test_state = np.array([0.0])  # Start with 1D assumption
        try:
            test_grad = self.grad_log_target(test_state)
            if np.isscalar(test_grad):
                self._n_dim = 1
            else:
                test_grad = np.atleast_1d(test_grad)
                self._n_dim = len(test_grad)
        except Exception:
            # If 1D fails, try with small random vector
            test_state = rng.standard_normal(2)
            test_grad = self.grad_log_target(test_state)
            self._n_dim = len(np.atleast_1d(test_grad))

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
            "step_size": np.full((n_chains, n_samples), self.step_size),
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
            # Generate initial states using standard normal
            if self._n_dim == 1:
                return rng.standard_normal((n_chains, 1))
            else:
                return rng.standard_normal((n_chains, self._n_dim))

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

    def _proposal_mean(self, x: np.ndarray) -> np.ndarray:
        """Compute the mean of the Langevin proposal distribution."""
        return x + 0.5 * self.step_size**2 * self.grad_log_target(x)

    def _log_proposal_density(self, x_new: np.ndarray, x_old: np.ndarray) -> float:
        """Compute log density of the proposal distribution q(x_new | x_old)."""
        mean = self._proposal_mean(x_old)
        if self._n_dim == 1:
            var = self.step_size**2
            return -0.5 * ((x_new - mean) ** 2 / var + np.log(2 * np.pi * var))
        else:
            cov = self.step_size**2 * np.eye(self._n_dim)
            return multivariate_normal.logpdf(x_new, mean, cov)

    def _propose(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Generate a proposal using Langevin dynamics."""
        mean = self._proposal_mean(x)
        if self._n_dim == 1:
            noise = rng.normal(0, self.step_size)
            return mean + noise
        else:
            noise = rng.normal(0, self.step_size, size=self._n_dim)
            return mean + noise

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

        current_log_target = self.log_target(current_state)
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
            # Generate proposal using Langevin dynamics
            if self._n_dim == 1:
                proposed_state = self._propose(np.array([current_state]), rng)[0]
            else:
                proposed_state = self._propose(current_state, rng)

            # Calculate log acceptance ratio
            try:
                proposed_log_target = self.log_target(proposed_state)

                # MALA acceptance ratio includes proposal correction
                if self._n_dim == 1:
                    log_ratio = (
                        proposed_log_target
                        - current_log_target
                        + self._log_proposal_density(
                            np.array([current_state]), np.array([proposed_state])
                        )
                        - self._log_proposal_density(
                            np.array([proposed_state]), np.array([current_state])
                        )
                    )
                else:
                    log_ratio = (
                        proposed_log_target
                        - current_log_target
                        + self._log_proposal_density(current_state, proposed_state)
                        - self._log_proposal_density(proposed_state, current_state)
                    )

                # Accept or reject
                if np.log(rng.random()) < log_ratio:
                    current_state = proposed_state
                    current_log_target = proposed_log_target
                    accepted = True
                else:
                    accepted = False

            except (ValueError, OverflowError, ZeroDivisionError):
                # Reject proposals that cause numerical issues
                accepted = False
                proposed_log_target = -np.inf

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
                    float(current_log_target)
                    if np.ndim(current_log_target) == 0
                    else float(current_log_target[0])
                )
                sample_stats["accepted"][chain_idx, sample_idx] = accepted
                sample_stats["step_size"][chain_idx, sample_idx] = self.step_size

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
            algorithm_name="Metropolis-adjusted Langevin Algorithm (MALA)",
            burn_in=burn_in,
            thin=thin,
            step_size=self.step_size,
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


def mala(
    log_target: Callable[[Union[float, np.ndarray]], float],
    grad_log_target: Callable[[Union[float, np.ndarray]], np.ndarray],
    n_samples: int,
    step_size: float = 0.1,
    initial_value: Union[float, np.ndarray] = 0.0,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """
    Metropolis-adjusted Langevin Algorithm sampler (functional interface).

    This is a simple functional implementation of the MALA algorithm.
    For more advanced features like multiple chains, ArviZ integration,
    and comprehensive diagnostics, use the MALASampler class.

    Parameters
    ----------
    log_target : callable
        Function that computes log π(x) for the target distribution.
        Should accept a single argument (the state) and return a float.
    grad_log_target : callable
        Function that computes the gradient ∇log π(x) of the log target density.
        Should return an array of the same shape as the input.
    n_samples : int
        Number of samples to generate from the target distribution.
    step_size : float, default=0.1
        Step size parameter ε for the Langevin dynamics.
    initial_value : float or array-like, default=0.0
        Starting value for the Markov chain. Should be in the support
        of the target distribution.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    samples : ndarray
        Array of shape (n_samples,) or (n_samples, n_dim) containing samples
        from the target distribution.
    accepted : int
        Number of accepted proposals out of n_samples total proposals.
        Acceptance rate can be computed as accepted / n_samples.

    Notes
    -----
    The MALA algorithm uses gradient information to propose new states:

    x' = x + (ε²/2) * ∇log π(x) + ε * Z

    where ε is the step size, π is the target distribution, and Z ~ N(0, I).
    The acceptance probability includes correction for the asymmetric proposal.

    Examples
    --------
    Sample from a standard normal distribution:

    >>> import numpy as np
    >>>
    >>> def log_normal(x):
    ...     return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)
    >>>
    >>> def grad_log_normal(x):
    ...     return -x
    >>>
    >>> samples, n_accepted = mala(
    ...     log_normal, grad_log_normal, 1000, step_size=0.5
    ... )
    >>> print(f"Acceptance rate: {n_accepted / 1000:.2f}")
    >>> print(f"Sample mean: {np.mean(samples):.3f}")
    >>> print(f"Sample std: {np.std(samples):.3f}")

    See Also
    --------
    MALASampler : Class-based implementation with more features
    """
    rng = as_generator(random_seed)

    # Determine dimensionality
    initial_value = np.atleast_1d(initial_value)
    n_dim = len(initial_value)

    samples = np.zeros((n_samples, n_dim) if n_dim > 1 else n_samples)
    current = initial_value.copy()
    accepted = 0

    for i in range(n_samples):
        # Langevin proposal
        grad = grad_log_target(current)
        mean = current + 0.5 * step_size**2 * grad

        if n_dim == 1:
            noise = rng.normal(0, step_size)
            proposed = mean + noise
        else:
            noise = rng.normal(0, step_size, size=n_dim)
            proposed = mean + noise

        # Acceptance ratio with proposal correction
        try:
            # Target ratio
            log_ratio = log_target(proposed) - log_target(current)

            # Proposal correction
            grad_proposed = grad_log_target(proposed)
            mean_reverse = proposed + 0.5 * step_size**2 * grad_proposed

            if n_dim == 1:
                log_ratio += (
                    -0.5
                    * ((current - mean_reverse) ** 2 - (proposed - mean) ** 2)
                    / step_size**2
                )
            else:
                diff_reverse = current - mean_reverse
                diff_forward = proposed - mean
                log_ratio += (
                    -0.5
                    * (np.sum(diff_reverse**2) - np.sum(diff_forward**2))
                    / step_size**2
                )

            # Accept or reject
            if np.log(rng.random()) < log_ratio:
                current = proposed
                accepted += 1

        except (ValueError, OverflowError):
            # Reject if numerical issues
            pass

        if n_dim == 1:
            samples[i] = current[0]
        else:
            samples[i] = current

    return samples, accepted
