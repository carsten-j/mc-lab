"""Hamiltonian Monte Carlo (HMC) sampler with automatic gradients.

This module implements the Hamiltonian Monte Carlo algorithm using PyTorch autograd
to automatically compute gradients. HMC uses Hamiltonian dynamics to propose new
states, allowing for efficient exploration of complex target distributions.

The implementation uses the leapfrog integrator for discretizing Hamiltonian dynamics
and prioritizes educational clarity while maintaining correctness.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Dict, List, Optional

import arviz as az
import numpy as np
import torch
from tqdm.auto import tqdm

from ._inference_data import create_inference_data
from ._rng import as_generator

__all__ = [
    "HMCSampler",
]


class HMCSampler:
    """
    Hamiltonian Monte Carlo (HMC) sampler with automatic gradients.

    This sampler uses PyTorch's automatic differentiation to compute gradients
    of the log target density, eliminating the need for analytical gradient
    functions. HMC uses Hamiltonian dynamics to propose new states via the
    leapfrog integrator.

    The algorithm simulates Hamiltonian dynamics:
    H(q, p) = U(q) + K(p) = -log π(q) + (1/2)p^T p

    where q is position (parameter space), p is momentum, U is potential energy,
    and K is kinetic energy. The leapfrog integrator provides a reversible,
    volume-preserving discretization of these dynamics.

    Parameters
    ----------
    log_target : callable
        Function that computes log π(x) for the target distribution.
        Should handle PyTorch tensors and return a scalar tensor.
        Must accept a single argument (the state tensor) and return a scalar.
    step_size : float, default=0.1
        Step size ε for the leapfrog integrator.
        Controls the discretization of Hamiltonian dynamics.
        Smaller values are more accurate but require more steps.
    n_leapfrog_steps : int, default=10
        Number of leapfrog steps L to take per HMC proposal.
        Total trajectory length is ε * L.
    var_names : list of str, optional
        Names for the sampled variables. Auto-generated if None.
        For multidimensional problems, should have length equal to dimension.

    Attributes
    ----------
    log_target : callable
        The target log probability density function.
    step_size : float
        The step size for the leapfrog integrator.
    n_leapfrog_steps : int
        The number of leapfrog steps per proposal.
    var_names : list of str
        Variable names for the sampled parameters.

    Examples
    --------
    Sample from a 1D standard normal distribution:

    >>> import torch
    >>> def log_normal(x):
    ...     return -0.5 * x**2
    >>>
    >>> sampler = HMCSampler(
    ...     log_target=log_normal,
    ...     step_size=0.2,
    ...     n_leapfrog_steps=10
    ... )
    >>> idata = sampler.sample(n_samples=1000, n_chains=4)

    Sample from a 2D correlated normal:

    >>> mean = torch.tensor([0.0, 0.0])
    >>> cov = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
    >>> cov_inv = torch.inverse(cov)
    >>>
    >>> def log_target(x):
    ...     diff = x - mean
    ...     return -0.5 * torch.sum(diff * (cov_inv @ diff))
    >>>
    >>> sampler = HMCSampler(
    ...     log_target=log_target,
    ...     step_size=0.15,
    ...     n_leapfrog_steps=20,
    ...     var_names=['x', 'y']
    ... )
    >>> idata = sampler.sample(n_samples=2000, burn_in=500)

    Notes
    -----
    **Requirements:**

    This sampler requires PyTorch for automatic differentiation.

    **Algorithm Overview:**

    HMC combines Hamiltonian dynamics with MCMC:

    1. **Momentum Sampling**: Sample p ~ N(0, I)
    2. **Leapfrog Integration**: Simulate Hamiltonian dynamics for L steps
       - p = p - (ε/2) * ∇U(q)  [half step momentum]
       - For i in range(L-1):
         - q = q + ε * p  [full step position]
         - p = p - ε * ∇U(q)  [full step momentum]
       - q = q + ε * p  [full step position]
       - p = p - (ε/2) * ∇U(q)  [half step momentum]
    3. **Negate momentum**: p* = -p* (ensures reversibility)
    4. **Metropolis acceptance**: Accept with probability min(1, exp(H₀ - H*))

    **Advantages over Random Walk:**

    - More efficient exploration via gradient information
    - Can make large moves while maintaining high acceptance
    - Reduces random walk behavior in complex distributions
    - Typical acceptance rates: 60-90% (higher than MALA)

    **Advantages over MALA:**

    - Second-order dynamics (position + momentum)
    - No proposal correction needed (symmetric proposals)
    - Can take longer trajectories for better exploration
    - More robust to step size choice

    **Tuning Guidelines:**

    - **Step size (ε)**: Balance between accuracy and efficiency
      - Too small: wasteful computation
      - Too large: numerical instability, low acceptance
    - **Number of steps (L)**: Controls trajectory length
      - Longer trajectories explore further
      - But increase computation per proposal
    - **Target acceptance**: 60-90% is typical for HMC
    - **Trajectory length**: ε * L should be comparable to typical scales

    References
    ----------
    .. [1] Neal, R. M. (2011). "MCMC using Hamiltonian dynamics."
           Handbook of Markov Chain Monte Carlo, 2(11), 2.
    .. [2] Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian
           Monte Carlo." arXiv:1701.02434.
    """

    def __init__(
        self,
        log_target: Callable[[torch.Tensor], torch.Tensor],
        step_size: float = 0.1,
        n_leapfrog_steps: int = 10,
        var_names: Optional[List[str]] = None,
    ):
        self.log_target = log_target
        self.step_size = step_size
        self.n_leapfrog_steps = n_leapfrog_steps
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

    def _hamiltonian(self, q: np.ndarray, p: np.ndarray) -> float:
        """Compute Hamiltonian H(q,p) = U(q) + K(p)."""
        # Potential energy: U(q) = -log π(q)
        U = -self._log_target_numpy(q)
        # Kinetic energy: K(p) = (1/2) p^T p
        K = 0.5 * np.sum(p**2)
        return U + K

    def _leapfrog(self, q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Leapfrog integrator for Hamiltonian dynamics."""
        q = q.copy()
        p = p.copy()

        # Half step for momentum
        grad = self._grad_log_target_numpy(q)
        p = p + 0.5 * self.step_size * grad

        # Full steps
        for _ in range(self.n_leapfrog_steps - 1):
            # Full step for position
            q = q + self.step_size * p
            # Full step for momentum
            grad = self._grad_log_target_numpy(q)
            p = p + self.step_size * grad

        # Final full step for position
        q = q + self.step_size * p

        # Half step for momentum
        grad = self._grad_log_target_numpy(q)
        p = p + 0.5 * self.step_size * grad

        return q, p

    def _propose(
        self, q: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, bool]:
        """Generate HMC proposal using leapfrog integration."""
        # Sample momentum
        p = rng.standard_normal(q.shape)

        # Compute current Hamiltonian
        current_H = self._hamiltonian(q, p)

        # Leapfrog integration
        try:
            q_new, p_new = self._leapfrog(q, p)

            # Negate momentum for reversibility
            p_new = -p_new

            # Compute proposed Hamiltonian
            proposed_H = self._hamiltonian(q_new, p_new)

            # Metropolis acceptance
            log_accept_prob = current_H - proposed_H

            if np.log(rng.random()) < log_accept_prob:
                return q_new, True
            else:
                return q, False

        except (ValueError, OverflowError, ZeroDivisionError):
            # Reject on numerical errors
            return q, False

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
        Generate samples using Hamiltonian Monte Carlo.

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
        The HMC algorithm proceeds as follows for each chain:

        1. Initialize chain at starting state
        2. For each iteration:
           a. Sample momentum p ~ N(0, I)
           b. Compute current Hamiltonian H₀ = H(q, p)
           c. Run leapfrog integrator for L steps
           d. Negate momentum: p* = -p*
           e. Compute proposed Hamiltonian H* = H(q*, p*)
           f. Accept with probability min(1, exp(H₀ - H*))
           g. Store sample if past burn-in and at thinning interval

        **Performance Tips:**

        - Use burn_in ≥ 1000 for complex distributions
        - Monitor acceptance rates: 60-90% is good for HMC
        - Tune step_size and n_leapfrog_steps together
        - Total trajectory length (step_size * n_leapfrog_steps) matters most
        """
        rng = as_generator(random_seed)

        # Determine dimensionality
        if self.var_names is not None:
            test_dims = [len(self.var_names)] + [1, 2, 3, 4, 5]
        else:
            test_dims = [1, 2, 3, 4, 5]

        for test_dim in test_dims:
            try:
                if test_dim == 1:
                    test_state = torch.tensor([0.0])
                else:
                    test_state = torch.tensor([0.0] * test_dim)

                self.log_target(test_state)
                test_grad = self._compute_gradient(test_state)
                self._n_dim = test_grad.numel()

                if self._n_dim == test_dim:
                    break

            except Exception as e:
                if test_dim == test_dims[-1]:
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
            "n_leapfrog_steps": np.full((n_chains, n_samples), self.n_leapfrog_steps),
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
            # HMC proposal
            if self._n_dim == 1:
                proposed_state, accepted = self._propose(np.array([current_state]), rng)
                proposed_state = proposed_state[0]
            else:
                proposed_state, accepted = self._propose(current_state, rng)

            if accepted:
                current_state = proposed_state
                current_log_target = self._log_target_numpy(current_state)

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
                sample_stats["n_leapfrog_steps"][chain_idx, sample_idx] = (
                    self.n_leapfrog_steps
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
            algorithm_name="Hamiltonian Monte Carlo (HMC)",
            burn_in=burn_in,
            thin=thin,
            step_size=self.step_size,
            n_leapfrog_steps=self.n_leapfrog_steps,
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
