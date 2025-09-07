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

    Returns
    -------
    estimate : float
        Estimated value of the integral E_p[f(x)].
    diagnostics : dict
        Dictionary containing diagnostic information:
        - 'weights': Normalized importance weights
        - 'log_weights': Log importance weights (unnormalized)
        - 'effective_sample_size': ESS estimate
        - 'cv_weights': Coefficient of variation of weights
        - 'max_weight': Maximum normalized weight
        - 'samples': Samples drawn from proposal
        - 'function_values': f(x) evaluated at samples
        - 'variance': Estimated variance of the estimator
        - 'standard_error': Standard error of the estimate
        - 'weight_entropy': Entropy of normalized weights
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

    # Coefficient of variation of weights
    if np.mean(weights_for_diagnostics) > 0:
        cv_weights = np.std(weights_for_diagnostics) / np.mean(weights_for_diagnostics)
    else:
        cv_weights = np.inf
    diagnostics["cv_weights"] = cv_weights

    # Maximum weight (indicates potential problems if too large)
    diagnostics["max_weight"] = np.max(weights_for_diagnostics)

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
    diagnostics["standard_error"] = np.sqrt(var_estimate)

    # Weight entropy (higher is better, max is log(n))
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    weight_entropy = -np.sum(
        weights_for_diagnostics * np.log(weights_for_diagnostics + epsilon)
    )
    diagnostics["weight_entropy"] = weight_entropy
    diagnostics["relative_entropy"] = weight_entropy / np.log(
        len(weights_for_diagnostics)
    )

    # Add warnings for poor performance
    if ess < 0.1 * n_samples:
        warnings.warn(
            f"Low ESS: {ess:.0f} ({100 * ess / n_samples:.1f}% efficiency). "
            "Consider using a better proposal distribution."
        )

    if diagnostics["max_weight"] > 0.1:
        warnings.warn(
            f"Maximum weight is {diagnostics['max_weight']:.3f}. "
            "Weight distribution is highly skewed."
        )

    return estimate, diagnostics


def diagnostic_plots(diagnostics: Dict, title: str = "Importance Sampling Diagnostics"):
    """
    Create diagnostic plots for importance sampling results.

    Parameters
    ----------
    diagnostics : dict
        Diagnostics dictionary from importance_sampling function.
    title : str
        Title for the plot.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Plot 1: Weight distribution
    ax = axes[0, 0]
    weights = diagnostics["weights"]
    ax.hist(weights, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(
        1 / len(weights),
        color="red",
        linestyle="--",
        label=f"Uniform weight (1/{len(weights):.0f})",
    )
    ax.set_xlabel("Normalized Weight")
    ax.set_ylabel("Frequency")
    ax.set_title("Weight Distribution")
    ax.legend()
    ax.set_yscale("log")

    # Plot 2: Cumulative weight distribution
    ax = axes[0, 1]
    sorted_weights = np.sort(weights)[::-1]
    cumsum_weights = np.cumsum(sorted_weights)
    ax.plot(range(1, len(weights) + 1), cumsum_weights)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.axhline(0.9, color="orange", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Cumulative Weight")
    ax.set_title("Cumulative Weight Distribution")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Plot 3: Log weights over iterations
    ax = axes[1, 0]
    log_weights = diagnostics["log_weights"]
    ax.plot(log_weights, alpha=0.6)
    ax.axhline(
        np.mean(log_weights),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(log_weights):.2f}",
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Log Weight")
    ax.set_title("Log Weights Over Samples")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = f"""
    ESS: {diagnostics["effective_sample_size"]:.0f}
    Efficiency: {diagnostics["proposal_efficiency"] * 100:.1f}%
    CV of weights: {diagnostics["cv_weights"]:.3f}
    Max weight: {diagnostics["max_weight"]:.4f}
    Weight entropy: {diagnostics["weight_entropy"]:.3f}
    Relative entropy: {diagnostics["relative_entropy"]:.3f}
    Standard Error: {diagnostics["standard_error"]:.4e}
    """

    ax.text(
        0.1,
        0.5,
        summary_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
    )
    ax.set_title("Summary Statistics")

    plt.tight_layout()
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Normal target, normal proposal
    print("Example 1: Computing E[X^2] under Normal(2, 1)")
    print("-" * 50)

    target = stats.norm(loc=2, scale=1)
    proposal = stats.norm(loc=1.5, scale=1.5)

    def f(x):
        return x**2

    estimate, diag = importance_sampling(f, target, proposal, n_samples=10000, seed=42)

    # True value: Var(X) + E[X]^2 = 1 + 4 = 5
    true_value = 1 + 2**2
    print(f"Estimate: {estimate:.4f}")
    print(f"True value: {true_value}")
    print(f"Error: {abs(estimate - true_value):.4f}")
    print(f"ESS: {diag['effective_sample_size']:.0f}")
    print(f"Efficiency: {diag['proposal_efficiency'] * 100:.1f}%")
    print()

    diagnostic_plots(diag)

    # Example 2: Unnormalized target distribution
    print("Example 2: Unnormalized Gaussian target")
    print("-" * 50)

    class UnnormalizedGaussian(stats.rv_continuous):
        def __init__(self, mu, sigma):
            super().__init__()
            self.mu = mu
            self.sigma = sigma

        def _pdf(self, x):
            # Returns unnormalized density (missing 1/sqrt(2π)σ factor)
            return np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

        def _logpdf(self, x):
            return -0.5 * ((x - self.mu) / self.sigma) ** 2

    unnorm_target = UnnormalizedGaussian(mu=3, sigma=0.5)
    proposal = stats.norm(loc=3, scale=1)

    def f(x):
        return x  # Computing E[X]

    estimate, diag = importance_sampling(
        f, unnorm_target, proposal, n_samples=10000, normalized=False, seed=42
    )

    print(f"Estimate E[X]: {estimate:.4f}")
    print("True value: 3.0")
    print(f"Error: {abs(estimate - 3.0):.4f}")
    print(f"ESS: {diag['effective_sample_size']:.0f}")
    print(f"Max weight: {diag['max_weight']:.4f}")
