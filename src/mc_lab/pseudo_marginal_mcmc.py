from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm


class PseudoMarginalMCMC:
    """
    Pseudo-marginal MCMC algorithm implementation based on slide 30.
    Uses log-space for numerical stability.

    The pseudo-marginal approach allows MCMC sampling when the target density
    can only be estimated unbiasedly (e.g., via particle filters or importance sampling).

    See also blog post:
    https://lips.cs.princeton.edu/pseudo-marginal-mcmc/
    https://darrenjw.wordpress.com/2010/09/20/the-pseudo-marginal-approach-to-exact-approximate-mcmc-algorithms/
    and:

    Key References:
    ---------------
    - Andrieu, C., & Roberts, G. O. (2009). The pseudo-marginal approach for
      efficient Monte Carlo computations. The Annals of Statistics, 37(2), 697-725.
      [Foundational paper establishing theoretical framework]

    - Beaumont, M. A. (2003). Estimation of population growth or decline in
      genetically monitored populations. Genetics, 164(3), 1139-1160.
      [Early application in population genetics]

    - Doucet, A., Pitt, M. K., Deligiannidis, G., & Kohn, R. (2015).
      Efficient implementation of Markov chain Monte Carlo when using an
      unbiased likelihood estimator. Biometrika, 102(2), 295-313.
      [Variance reduction techniques and efficiency improvements]

    - Sherlock, C., Thiery, A. H., Roberts, G. O., & Rosenthal, J. S. (2015).
      On the efficiency of pseudo-marginal random walk Metropolis algorithms.
      The Annals of Statistics, 43(1), 238-275.
      [Theoretical analysis of efficiency and optimal tuning]
    """

    def __init__(
        self,
        log_unbiased_estimator: Callable[[np.ndarray], Tuple[float, Any]],
        proposal_sampler: Callable[[np.ndarray], np.ndarray],
        log_proposal_density: Callable[[np.ndarray, np.ndarray], float],
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters:
        -----------
        log_unbiased_estimator : function
            Function that returns log of an unbiased estimate of π(x)
            Should return (log_estimate, auxiliary_info) tuple
        proposal_sampler : function
            Function q(x, ·) that samples a proposal given current state x
        log_proposal_density : function
            Function that evaluates log q(y|x) for acceptance ratio
        seed : int, optional
            Random seed for reproducible acceptance decisions in step 4
        """
        self.log_unbiased_estimator = log_unbiased_estimator
        self.proposal_sampler = proposal_sampler
        self.log_proposal_density = log_proposal_density
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def sample(
        self, x0: Union[float, np.ndarray], n_samples: int, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the pseudo-marginal MCMC algorithm.

        Parameters:
        -----------
        x0 : initial state
        n_samples : number of samples to generate
        verbose : whether to print progress

        Returns:
        --------
        samples : array of samples
        log_estimates : array of log π̂ values (for diagnostics)
        acceptance_rate : proportion of accepted proposals
        """
        samples = np.zeros((n_samples, len(np.atleast_1d(x0))))
        log_estimates = np.zeros(n_samples)

        # Initialize
        x_current = np.atleast_1d(x0)
        log_pi_hat_current, _ = self.log_unbiased_estimator(x_current)

        samples[0] = x_current
        log_estimates[0] = log_pi_hat_current
        n_accepted = 0

        progress_bar = tqdm(
            range(1, n_samples), desc="MCMC Sampling", disable=not verbose
        )

        for t in progress_bar:
            # Step 1: Propose Y ~ q(x, ·)
            y_proposed = self.proposal_sampler(x_current)

            # Step 2: Sample log π̂(Y) - get log of unbiased estimate at proposed point
            log_pi_hat_proposed, _ = self.log_unbiased_estimator(y_proposed)

            # Step 3: Compute log acceptance probability
            # log α = min{0, log[π̂(Y)] + log[q(Y,X^(t-1))] - log[π̂(X^(t-1))] - log[q(X^(t-1),Y)]}
            log_q_forward = self.log_proposal_density(
                y_proposed, x_current
            )  # log q(Y|X)
            log_q_backward = self.log_proposal_density(
                x_current, y_proposed
            )  # log q(X|Y)

            log_acceptance_ratio = (
                log_pi_hat_proposed
                + log_q_backward
                - log_pi_hat_current
                - log_q_forward
            )

            log_alpha = min(0.0, log_acceptance_ratio)

            # Step 4: Accept or reject (still need to convert to probability for comparison)
            if np.log(self.rng.random()) < log_alpha:
                x_current = y_proposed
                log_pi_hat_current = log_pi_hat_proposed
                n_accepted += 1

            samples[t] = x_current
            log_estimates[t] = log_pi_hat_current

            # Update progress bar with acceptance rate
            if (
                t % 100 == 0
            ):  # Update every 100 iterations to avoid too frequent updates
                progress_bar.set_postfix({"Accept Rate": f"{n_accepted / t:.2%}"})

        return samples, log_estimates, n_accepted / n_samples
