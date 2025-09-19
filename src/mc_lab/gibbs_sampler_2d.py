from collections.abc import Callable
from typing import Dict, Optional, Tuple

import arviz as az
import numpy as np
from tqdm.auto import tqdm

from ._inference_data import create_inference_data


class GibbsSampler2D:
    """
    Gibbs sampler for 2D joint distributions p(x,y) with ArviZ integration.

    This sampler implements the Gibbs sampling algorithm for bivariate distributions
    by alternately sampling from conditional distributions p(x|y) and p(y|x).
    The implementation includes comprehensive diagnostics through ArviZ integration
    and supports multiple chains for convergence assessment.

    Gibbs sampling is particularly effective for distributions where the conditional
    distributions are easy to sample from, even when the joint distribution is complex.
    It's guaranteed to converge to the correct target distribution under mild conditions.

    Parameters
    ----------
    sample_x_given_y : callable
        Function that samples from p(x|y). Takes y value, returns sampled x.
    sample_y_given_x : callable
        Function that samples from p(y|x). Takes x value, returns sampled y.
    log_target : callable, optional
        Function that computes log p(x,y). Used for diagnostics.
    var_names : tuple of str, default=("x", "y")
        Names for the two variables.

    Examples
    --------
    Sample from a bivariate normal distribution:

    >>> import numpy as np
    >>> from scipy import stats
    >>>
    >>> # Define conditional distributions for bivariate normal
    >>> def sample_x_given_y(y):
    ...     # x | y ~ N(rho * y, 1 - rho^2) for rho = 0.5
    ...     return np.random.normal(0.5 * y, np.sqrt(0.75))
    >>>
    >>> def sample_y_given_x(x):
    ...     # y | x ~ N(rho * x, 1 - rho^2) for rho = 0.5
    ...     return np.random.normal(0.5 * x, np.sqrt(0.75))
    >>>
    >>> def log_joint(x, y):
    ...     # Log density of bivariate normal with correlation 0.5
    ...     cov_inv = np.array([[4/3, -2/3], [-2/3, 4/3]])
    ...     vec = np.array([x, y])
    ...     return -0.5 * vec @ cov_inv @ vec - np.log(2 * np.pi * np.sqrt(0.75))
    >>>
    >>> sampler = GibbsSampler2D(
    ...     sample_x_given_y, sample_y_given_x, log_joint, var_names=["x", "y"]
    ... )
    >>> idata = sampler.sample(n_samples=1000, n_chains=4)
    >>> print(az.summary(idata))

    Notes
    -----
    **Algorithm Overview:**

    The Gibbs sampler alternates between sampling from conditional distributions:

    1. Sample x^(t+1) ~ p(x | y^(t))
    2. Sample y^(t+1) ~ p(y | x^(t+1))
    3. Repeat until convergence

    **Convergence Properties:**

    - Gibbs sampling always has acceptance rate 1 (no rejections)
    - Convergence depends on correlation between variables
    - Highly correlated variables may lead to slow mixing
    - Multiple chains help assess convergence

    **When to Use Gibbs Sampling:**

    - When conditional distributions are easy to sample from
    - For hierarchical models with conjugate priors
    - When Metropolis-Hastings would have low acceptance rates
    - For high-dimensional problems with conditional independence

    References
    ----------
    .. [1] Geman, S., & Geman, D. (1984). "Stochastic relaxation, Gibbs distributions,
           and the Bayesian restoration of images." IEEE transactions on pattern
           analysis and machine intelligence, (6), 721-741.
    .. [2] Casella, G., & George, E. I. (1992). "Explaining the Gibbs sampler."
           The American Statistician, 46(3), 167-174.
    """

    def __init__(
        self,
        sample_x_given_y: Callable[[float], float],
        sample_y_given_x: Callable[[float], float],
        log_target: Optional[Callable[[float, float], float]] = None,
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
        log_target : callable, optional
            Function that computes log p(x,y). Used for diagnostics.
        var_names : tuple of str
            Names for the two variables (default: 'x', 'y')
        """
        self.sample_x_given_y = sample_x_given_y
        self.sample_y_given_x = sample_y_given_x
        self.log_target = log_target
        self.var_names = var_names

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
        initial_states : np.ndarray, optional
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
        if initial_states is None:
            initial_states = np.random.randn(n_chains, 2)
        elif initial_states.shape == (2,):
            # Add small random perturbation to avoid identical chains
            initial_states = (
                initial_states[np.newaxis, :] + np.random.randn(n_chains, 2) * 0.1
            )
        elif initial_states.shape[0] != n_chains:
            raise ValueError(
                f"initial_states shape {initial_states.shape} doesn't match "
                f"n_chains {n_chains}"
            )

        # Storage for all chains
        total_iterations = burn_in + n_samples * thin
        posterior_samples = {
            self.var_names[0]: np.zeros((n_chains, n_samples)),
            self.var_names[1]: np.zeros((n_chains, n_samples)),
        }

        # Sample statistics storage
        sample_stats = {}
        if self.log_target is not None:
            sample_stats["log_likelihood"] = np.zeros((n_chains, n_samples))

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

                    # Compute sample statistics if log_target is provided
                    if self.log_target is not None:
                        log_likelihood = self.log_target(
                            current_state[0], current_state[1]
                        )
                        sample_stats["log_likelihood"][chain_idx, sample_idx] = (
                            log_likelihood
                        )

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

    def get_acceptance_rates(self, idata: az.InferenceData) -> Dict[str, float]:
        """
        Get acceptance rates from InferenceData.

        Note: Gibbs sampling always has 100% acceptance rate since proposals
        are always accepted. This method is provided for interface consistency
        with other MCMC samplers.

        Parameters
        ----------
        idata : arviz.InferenceData
            InferenceData from sampling.

        Returns
        -------
        rates : dict
            Acceptance rates (always 1.0 for Gibbs sampling).
        """
        n_chains = idata.posterior.sizes["chain"]

        rates = {"overall": 1.0}
        for chain in range(n_chains):
            rates[f"chain_{chain}"] = 1.0

        return rates

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
        return create_inference_data(
            posterior_samples=posterior_samples,
            sample_stats=sample_stats,
            n_chains=n_chains,
            n_samples=n_samples,
            n_dim=2,  # Gibbs sampler is always 2D
            algorithm_name="Gibbs Sampling",
            burn_in=burn_in,
            thin=thin,
        )
