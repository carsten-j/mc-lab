import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats


def importance_sampling(
    f: Callable[[np.ndarray], np.ndarray],
    target: stats.rv_continuous,
    proposal: stats.rv_continuous,
    n_samples: int = 10000,
    normalized: bool = True,
    seed: Optional[int] = None,
) -> Tuple[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Importance sampling algorithm to compute E_p[f(x)] = ∫ f(x)p(x)dx.

    Parameters
    ----------
    f : Callable
        Function to integrate. Should accept array input and return array output.
    target : scipy.stats.rv_continuous
        Target distribution p(x). Can be normalized or unnormalized.
    proposal : scipy.stats.rv_continuous
        Proposal distribution q(x) for sampling.
    n_samples : int, default=10000
        Number of samples to draw from proposal distribution.
    normalized : bool, default=True
        Whether the target distribution p is normalized (integrates to 1).
        If False, self-normalized importance sampling is used.
    seed : int, optional
        Random seed for reproducibility.

    For a discussion of diagnostics (especially ESS) see Art Owens book,
    chapter 9 about Importance Sampling. There is also a great blog post
    by Sebastian Nowozin about ESS, see
    https://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html

    ToDo: Implement Art Owen metrics that include the integrand f and average of weights.

    Returns
    -------
    estimate : float
        Estimated value of the integral E_p[f(x)].
    diagnostics : dict
        Dictionary containing diagnostic information:
        - 'weights': Normalized importance weights
        - 'log_weights': Log importance weights (unnormalized)
        - 'effective_sample_size': ESS estimate
        - 'samples': Samples drawn from proposal
        - 'function_values': f(x) evaluated at samples
        - 'variance': Estimated variance of the estimator
        - 'proposal_efficiency': ESS / n_samples

    Examples
    --------
    >>> # Example 1: Computing expectation of x^2 under normal distribution
    >>> target = stats.norm(loc=2, scale=1)
    >>> proposal = stats.norm(loc=0, scale=2)
    >>> f = lambda x: x**2
    >>> estimate, diagnostics = importance_sampling(f, target, proposal, n_samples=5000)
    >>> print(f"Estimate: {estimate:.4f}")
    >>> print(f"ESS: {diagnostics['effective_sample_size']:.0f}")

    >>> # Example 2: Unnormalized target distribution
    >>> class UnnormalizedGaussian(stats.rv_continuous):
    ...     def __init__(self, mu, sigma):
    ...         super().__init__()
    ...         self.mu = mu
    ...         self.sigma = sigma
    ...     def _pdf(self, x):
    ...         return np.exp(-0.5 * ((x - self.mu) / self.sigma)**2)
    >>>
    >>> unnorm_target = UnnormalizedGaussian(mu=1, sigma=0.5)
    >>> proposal = stats.norm(loc=1, scale=1)
    >>> f = lambda x: x
    >>> estimate, diag = importance_sampling(f, unnorm_target, proposal,
    ...                                      n_samples=10000, normalized=False)
    """

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Draw samples from proposal distribution
    samples = proposal.rvs(size=n_samples)

    # Compute log densities for numerical stability
    log_p = target.logpdf(samples)
    log_q = proposal.logpdf(samples)

    # Check for invalid values
    if np.any(np.isnan(log_p)) or np.any(np.isnan(log_q)):
        warnings.warn(
            "NaN values detected in log densities. Check distribution support."
        )

    if np.any(np.isinf(log_q)):
        warnings.warn(
            "Infinite values in proposal log density. Some samples may be outside support."
        )

    # Compute log importance weights
    log_weights = log_p - log_q

    # Handle -inf values (samples outside target support)
    valid_idx = np.isfinite(log_weights)
    if not np.all(valid_idx):
        warnings.warn(
            f"{np.sum(~valid_idx)} samples have invalid weights (outside support)."
        )
        log_weights = log_weights[valid_idx]
        samples_valid = samples[valid_idx]
    else:
        samples_valid = samples

    # For numerical stability, subtract max log weight before exponentiating
    max_log_weight = np.max(log_weights)
    weights = np.exp(log_weights - max_log_weight)

    # Normalize weights
    if normalized:
        # Standard importance sampling (target is normalized)
        weights_normalized = weights * np.exp(max_log_weight)
        # Ensure normalization for diagnostics
        weights_for_diagnostics = weights_normalized / np.sum(weights_normalized)
    else:
        # Self-normalized importance sampling (target is unnormalized)
        weights_normalized = weights / np.sum(weights)
        weights_for_diagnostics = weights_normalized

    # Evaluate function at samples
    f_values = f(samples_valid)

    # Compute estimate
    if normalized:
        estimate = np.mean(f_values * weights_normalized)
    else:
        estimate = np.sum(f_values * weights_normalized)

    # Compute diagnostics
    diagnostics = {}

    # Store basic quantities
    diagnostics["weights"] = weights_for_diagnostics
    diagnostics["log_weights"] = log_weights
    diagnostics["samples"] = samples_valid
    diagnostics["function_values"] = f_values

    # Effective Sample Size (ESS)
    ess = 1.0 / np.sum(weights_for_diagnostics**2)
    diagnostics["effective_sample_size"] = ess
    diagnostics["proposal_efficiency"] = ess / n_samples

    # Variance estimation
    if normalized:
        # For standard IS with normalized target
        var_estimate = np.var(f_values * weights_normalized) / len(samples_valid)
    else:
        # For self-normalized IS
        # Using delta method approximation
        sum_weights = np.sum(weights)
        var_f = np.sum(weights * (f_values - estimate) ** 2) / sum_weights
        var_estimate = var_f / ess

    diagnostics["variance"] = var_estimate

    # Add warnings for poor performance
    if ess < 0.1 * n_samples:
        warnings.warn(
            f"Low ESS: {ess:.0f} ({100 * ess / n_samples:.1f}% efficiency). "
            "Consider using a better proposal distribution."
        )

    return estimate, diagnostics
