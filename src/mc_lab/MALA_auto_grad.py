"""Metropolis-adjusted Langevin Algorithm (MALA) MCMC sampler with automatic gradients.

This module implements the MALA algorithm using PyTorch autograd to automatically
compute gradients, eliminating the need for users to provide analytical gradients.
This makes MALA more accessible while maintaining the efficiency of gradient-based
proposals.

The implementation prioritizes educational clarity and includes full ArviZ integration
for comprehensive MCMC diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Dict, List, Optional

import arviz as az
import numpy as np
import torch
from scipy.stats import multivariate_normal
from tqdm.auto import tqdm

from ._inference_data import create_inference_data
from ._rng import as_generator

__all__ = [
    "MALAAutoGradSampler",
]


class MALAAutoGradSampler:
    """
    Metropolis-adjusted Langevin Algorithm (MALA) sampler with automatic gradients.

    This sampler uses PyTorch's automatic differentiation to compute gradients
    of the log target density, eliminating the need for users to provide analytical
    gradient functions. This makes MALA more accessible while maintaining the
    efficiency of gradient-based proposals.

    The algorithm uses Langevin dynamics to propose new states:
    x' = x + (ε²/2) * ∇log π(x) + ε * Z

    where ε is the step size, π is the target distribution, and Z ~ N(0, I).
    This gradient-guided proposal allows for more efficient exploration of the
    target distribution compared to random walk methods.

    Parameters
    ----------
    log_target : callable
        Function that computes log π(x) for the target distribution.
        Should handle PyTorch tensors and return a scalar tensor.
        Must accept a single argument (the state tensor) and return a scalar.
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
    step_size : float
        The step size parameter for Langevin dynamics.
    var_names : list of str
        Variable names for the sampled parameters.

    Examples
    --------
    Sample from a 2D multivariate normal distribution:

    >>> import torch
    >>> import numpy as np
    >>>
    >>> # Define target distribution (2D normal with correlation)
    >>> mean = torch.tensor([0.0, 0.0])
    >>> cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    >>> cov_inv = torch.inverse(cov)
    >>>
    >>> def log_target(x):
    ...     diff = x - mean
    ...     return -0.5 * torch.sum(diff * (cov_inv @ diff))
    >>>
    >>> sampler = MALAAutoGradSampler(
    ...     log_target=log_target,
    ...     step_size=0.2,
    ...     var_names=['x', 'y']
    ... )
    >>> idata = sampler.sample(n_samples=1000, n_chains=4)

    Sample from a 1D distribution:

    >>> def log_normal(x):
    ...     return -0.5 * x**2 - 0.5 * torch.log(torch.tensor(2 * torch.pi))
    >>>
    >>> sampler = MALAAutoGradSampler(
    ...     log_target=log_normal,
    ...     step_size=0.5
    ... )
    >>> idata = sampler.sample(n_samples=2000, burn_in=500)

    Notes
    -----
    **Requirements:**

    This sampler requires PyTorch for automatic differentiation.
    Install with: pip install torch

    **Algorithm Overview:**

    MALA combines the efficiency of gradient-based methods with the theoretical
    guarantees of MCMC:

    1. **Automatic Gradients**: Uses PyTorch autograd to compute ∇log π(x)
    2. **Proposal Generation**: Use Langevin dynamics to propose new states
       x' = x + (ε²/2) * ∇log π(x) + ε * Z, where Z ~ N(0, I)
    3. **Acceptance Step**: Accept/reject using Metropolis criterion with
       asymmetric proposal correction

    **Advantages over Manual Gradients:**

    - No need to derive analytical gradients
    - Reduces implementation errors
    - Easier to experiment with complex target distributions
    - Automatic handling of gradient computation edge cases

    **Performance Considerations:**

    - Slightly slower than analytical gradients due to autograd overhead
    - Memory usage may be higher for complex functions
    - Still much more efficient than random walk methods

    **Tuning Guidelines:**

    - **Step size**: Controls trade-off between acceptance rate and step size
    - Target acceptance rate: 50-70% (higher than random walk MH)
    - Too small ε: High acceptance but inefficient exploration
    - Too large ε: Low acceptance due to overshooting

    References
    ----------
    .. [1] Roberts, G. O., & Tweedie, R. L. (1996). "Exponential convergence of
           Langevin distributions and their discrete approximations."
           Bernoulli, 2(4), 341-363.
    .. [2] Paszke, A., et al. (2019). "PyTorch: An imperative style,
           high-performance deep learning library." NeurIPS.
    """

    def __init__(
        self,
        log_target: Callable[[torch.Tensor], torch.Tensor],
        step_size: float = 0.1,
        var_names: Optional[List[str]] = None,
    ):
        self.log_target = log_target
        self.step_size = step_size
        self.var_names = var_names

        # Will be set during sampling
        self._n_dim = None

    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient using PyTorch autograd."""
        x_tensor = x.clone().detach().requires_grad_(True)
        log_prob = self.log_target(x_tensor)
        grad = torch.autograd.grad(log_prob, x_tensor, create_graph=False)[0]
        return grad.detach()

    def _log_target_numpy(self, x: np.ndarray) -> float:
        """Wrapper to evaluate log target with numpy arrays."""
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            result = self.log_target(x_tensor)
        return float(result.item())

    def _grad_log_target_numpy(self, x: np.ndarray) -> np.ndarray:
        """Wrapper to compute gradient with numpy arrays."""
        x_tensor = torch.tensor(x, dtype=torch.float32)
        grad_tensor = self._compute_gradient(x_tensor)
        return grad_tensor.numpy()

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
        Generate samples using MALA with automatic gradients.

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
           a. Compute gradient ∇log π(x) using PyTorch autograd
           b. Propose new state using Langevin dynamics:
              x' = x + (ε²/2) * ∇log π(x) + ε * Z, where Z ~ N(0, I)
           c. Compute acceptance probability with proposal correction
           d. Accept/reject with computed probability
           e. Store sample if past burn-in and at thinning interval

        **Performance Tips:**

        - Use burn_in ≥ 1000 for complex distributions
        - Monitor acceptance rates: 50-70% is typically good for MALA
        - Tune step_size to achieve target acceptance rate
        - Consider using torch.jit.script for the log_target function for speed
        """
        rng = as_generator(random_seed)

        # Determine dimensionality by testing the log_target function
        # If var_names is provided, use its length as a hint
        if self.var_names is not None:
            test_dims = [len(self.var_names)] + [1, 2, 3, 4, 5]
        else:
            test_dims = [1, 2, 3, 4, 5]

        # Try different dimensions to find the correct one
        for test_dim in test_dims:
            try:
                if test_dim == 1:
                    test_state = torch.tensor([0.0])
                else:
                    test_state = torch.tensor([0.0] * test_dim)

                self.log_target(test_state)
                test_grad = self._compute_gradient(test_state)
                self._n_dim = test_grad.numel()

                # If we successfully computed gradient, we found the right dimension
                if self._n_dim == test_dim:
                    break

            except Exception as e:
                if test_dim == test_dims[-1]:  # Last attempt
                    raise RuntimeError(f"Could not determine dimensionality: {e}")
                continue

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
        grad = self._grad_log_target_numpy(x)
        return x + 0.5 * self.step_size**2 * grad

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

        current_log_target = self._log_target_numpy(current_state)
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
                proposed_log_target = self._log_target_numpy(proposed_state)

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

                sample_stats["log_likelihood"][chain_idx, sample_idx] = float(
                    current_log_target
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
            algorithm_name="MALA with Automatic Gradients",
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
