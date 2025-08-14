import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class Distribution(ABC):
    """Abstract base class for probability distributions"""

    @abstractmethod
    def sample(self, n_samples):
        """Generate n_samples from the distribution"""
        pass

    @abstractmethod
    def log_pdf(self, x):
        """Compute log probability density function"""
        pass

    def pdf(self, x):
        """Compute probability density function"""
        return np.exp(self.log_pdf(x))


class Normal(Distribution):
    """Normal distribution implementation"""

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def sample(self, n_samples):
        return np.random.normal(self.mean, self.std, n_samples)

    def log_pdf(self, x):
        return stats.norm.logpdf(x, self.mean, self.std)


class MixtureNormal(Distribution):
    """Mixture of two normal distributions"""

    def __init__(self, mean1, std1, mean2, std2, weight1=0.5):
        self.mean1, self.std1 = mean1, std1
        self.mean2, self.std2 = mean2, std2
        self.weight1 = weight1
        self.weight2 = 1 - weight1

    def sample(self, n_samples):
        # Sample from mixture
        n1 = np.random.binomial(n_samples, self.weight1)
        n2 = n_samples - n1

        samples1 = np.random.normal(self.mean1, self.std1, n1)
        samples2 = np.random.normal(self.mean2, self.std2, n2)

        samples = np.concatenate([samples1, samples2])
        np.random.shuffle(samples)
        return samples

    def log_pdf(self, x):
        pdf1 = stats.norm.pdf(x, self.mean1, self.std1)
        pdf2 = stats.norm.pdf(x, self.mean2, self.std2)
        mixture_pdf = self.weight1 * pdf1 + self.weight2 * pdf2
        return np.log(mixture_pdf + 1e-100)  # Add small constant to avoid log(0)


class NonStandardDistribution(Distribution):
    """Non-standard unnormalized distribution: exp(-x²/4) * (1 + sin²(2x))"""

    def __init__(self):
        # This distribution is unnormalized and difficult to sample from directly
        pass

    def sample(self, n_samples):
        # For this example, we don't implement direct sampling
        # In practice, you might use rejection sampling or MCMC
        raise NotImplementedError(
            "Direct sampling not implemented for this distribution"
        )

    def log_pdf(self, x):
        """Unnormalized log PDF"""
        return -(x**2) / 4 + np.log(1 + np.sin(2 * x) ** 2)

    def pdf(self, x):
        """Unnormalized PDF"""
        return np.exp(-(x**2) / 4) * (1 + np.sin(2 * x) ** 2)


class ImportanceSampler:
    """General importance sampling implementation"""

    def __init__(self, target_dist, proposal_dist, function, use_self_normalized=False):
        self.target_dist = target_dist
        self.proposal_dist = proposal_dist
        self.function = function
        self.use_self_normalized = use_self_normalized

    def estimate(self, n_samples, return_diagnostics=False, return_samples=False):
        """Perform importance sampling estimation"""
        samples = self.proposal_dist.sample(n_samples)
        weights = self.compute_weights(samples)
        function_values = self.function(samples)

        # Handle potential numerical issues
        if np.any(np.isinf(weights)) or np.any(np.isnan(weights)):
            warnings.warn(
                "Infinite or NaN weights detected. Results may be unreliable."
            )

        if self.use_self_normalized:
            # Self-normalized importance sampling (for expectations under unnormalized distributions)
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                raise ValueError(
                    "Sum of weights is zero. Proposal distribution may be inappropriate."
                )
            estimate = np.sum(weights * function_values) / weights_sum
        else:
            # Regular importance sampling (for normalizing constants, expectations under normalized distributions)
            estimate = np.mean(weights * function_values)

        results = {"estimate": estimate}

        if return_diagnostics:
            results["diagnostics"] = self.compute_diagnostics(weights)

        if return_samples:
            results["samples"] = samples
            results["weights"] = weights
            results["function_values"] = function_values

        return results

    def compute_weights(self, samples):
        """Compute importance weights w_i = p(x_i) / q(x_i)"""
        log_target = self.target_dist.log_pdf(samples)
        log_proposal = self.proposal_dist.log_pdf(samples)
        log_weights = log_target - log_proposal

        # Clip extreme log weights for numerical stability
        log_weights = np.clip(log_weights, -50, 50)

        return np.exp(log_weights)

    def compute_diagnostics(self, weights):
        """Compute diagnostic statistics for importance sampling"""
        weights = np.array(weights)
        weights_sum = np.sum(weights)
        weights_sq_sum = np.sum(weights**2)

        if weights_sum == 0:
            return {"effective_sample_size": 0, "warning": "All weights are zero"}

        # Effective sample size
        ess = weights_sum**2 / weights_sq_sum if weights_sq_sum > 0 else 0

        # Coefficient of variation of weights
        cv_weights = (
            np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else np.inf
        )

        return {
            "effective_sample_size": ess,
            "relative_ess": ess / len(weights),
            "cv_weights": cv_weights,
            "max_weight": np.max(weights),
            "min_weight": np.min(weights),
            "n_zero_weights": np.sum(weights == 0),
            "n_negligible_weights": np.sum(weights < 1e-10),
            "weight_concentration": np.sum(
                weights > np.mean(weights) + 2 * np.std(weights)
            ),
        }


def create_visualization(target_dist, proposal_dist, sampler_results, x_range=(-6, 6)):
    """Create comprehensive visualization of importance sampling results"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Importance Sampling Analysis: Non-standard PDF Integration", fontsize=16
    )

    x_plot = np.linspace(x_range[0], x_range[1], 1000)

    # Plot 1: Target and proposal distributions
    ax = axes[0, 0]
    target_pdf = target_dist.pdf(x_plot)
    proposal_pdf = proposal_dist.pdf(x_plot)

    ax.plot(x_plot, target_pdf, "b-", linewidth=2, label="Target (unnormalized)")
    ax.plot(x_plot, proposal_pdf, "r--", linewidth=2, label="Proposal")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Target vs Proposal Distributions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Samples and weights
    ax = axes[0, 1]
    samples = sampler_results["samples"]
    weights = sampler_results["weights"]

    # Color samples by weight magnitude
    scatter = ax.scatter(
        samples, np.zeros_like(samples), c=weights, s=30, alpha=0.6, cmap="viridis"
    )
    ax.set_xlabel("Sample values")
    ax.set_title("Samples colored by importance weights")
    ax.set_ylim(-0.1, 0.1)
    plt.colorbar(scatter, ax=ax, label="Weight")
    ax.grid(True, alpha=0.3)

    # Plot 3: Weight distribution
    ax = axes[0, 2]
    ax.hist(weights, bins=50, alpha=0.7, density=True, color="purple")
    ax.axvline(
        np.mean(weights),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(weights):.3f}",
    )
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Importance Weights")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Cumulative estimate convergence
    ax = axes[1, 0]
    function_values = sampler_results["function_values"]
    cumulative_estimate = np.cumsum(weights * function_values) / np.cumsum(weights)

    ax.plot(cumulative_estimate, "b-", linewidth=2)
    ax.axhline(
        cumulative_estimate[-1],
        color="red",
        linestyle="--",
        label=f"Final estimate: {cumulative_estimate[-1]:.4f}",
    )
    ax.set_xlabel("Sample number")
    ax.set_ylabel("Cumulative estimate")
    ax.set_title("Convergence of Integral Estimate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Effective sample size evolution
    ax = axes[1, 1]
    n_samples = len(weights)
    ess_evolution = []

    for i in range(100, n_samples, max(1, n_samples // 50)):
        w_partial = weights[:i]
        ess_partial = np.sum(w_partial) ** 2 / np.sum(w_partial**2)
        ess_evolution.append(ess_partial / i)  # Relative ESS

    x_ess = range(100, n_samples, max(1, n_samples // 50))
    ax.plot(x_ess, ess_evolution, "g-", linewidth=2)
    ax.set_xlabel("Sample number")
    ax.set_ylabel("Relative Effective Sample Size")
    ax.set_title("Evolution of Sampling Efficiency")
    ax.grid(True, alpha=0.3)

    # Plot 6: Diagnostics summary
    ax = axes[1, 2]
    ax.axis("off")

    diagnostics = sampler_results["diagnostics"]
    diag_text = f"""
    Importance Sampling Diagnostics:
    
    • Final Estimate: {sampler_results["estimate"]:.6f}
    • Total Samples: {len(samples):,}
    • Effective Sample Size: {diagnostics["effective_sample_size"]:.1f}
    • Relative ESS: {diagnostics["relative_ess"]:.3f}
    • CV of Weights: {diagnostics["cv_weights"]:.3f}
    • Max Weight: {diagnostics["max_weight"]:.3f}
    • Min Weight: {diagnostics["min_weight"]:.6f}
    • Zero Weights: {diagnostics["n_zero_weights"]}
    • Negligible Weights: {diagnostics["n_negligible_weights"]}
    
    Efficiency Assessment:
    • {
        "Excellent"
        if diagnostics["relative_ess"] > 0.5
        else "Good"
        if diagnostics["relative_ess"] > 0.1
        else "Poor"
        if diagnostics["relative_ess"] > 0.01
        else "Very Poor"
    }
    """

    ax.text(
        0.05,
        0.95,
        diag_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    return fig
