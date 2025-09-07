# Reusable visualization function for all distribution examples
from ast import Dict

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.gofplots import qqplot


def plot_distribution_analysis(
    samples,
    scipy_dist,
    title,
    x_range=None,
    n_points=1000,
):
    """
    Create a comprehensive 4-panel visualization for distribution analysis.

    Parameters:
    -----------
    samples : array-like
        Generated samples from the distribution
    distribution_name : str
        Name of the distribution (e.g., "Exponential", "Cauchy")
    example_number : str
        Example identifier (e.g., "2.1", "3.2")
    method_description : str
        Description of the sampling method used
    scipy_dist : scipy.stats distribution object
        Scipy distribution for comparison (with parameters set)
    x_range : tuple, optional
        (min, max) range for x-axis. If None, will be inferred from samples
    n_points : int, optional
        Number of points for theoretical curves (default: 1000)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        title,
        fontsize=16,
    )

    # Determine x-range for theoretical curves
    if x_range is None:
        x_min = np.min(samples)
        x_max = np.max(samples)
        # Add some padding
        padding = (x_max - x_min) * 0.1
        x_range = (max(0, x_min - padding), x_max + padding)

    x_theory = np.linspace(x_range[0], x_range[1], n_points)

    # Plot 1: PDF Comparison (Histogram vs Theoretical)
    pdf_theory = scipy_dist.pdf(x_theory)

    axes[0, 0].hist(
        samples,
        bins=50,
        density=True,
        alpha=0.7,
        label="Generated samples",
    )
    axes[0, 0].plot(
        x_theory,
        pdf_theory,
        "r-",
        linewidth=2,
        label="Theoretical PDF",
    )
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("PDF Comparison")
    axes[0, 0].legend()

    # Plot 2: CDF Comparison
    theoretical_cdf = scipy_dist.cdf(x_theory)
    empirical_cdf = np.searchsorted(np.sort(samples), x_theory) / len(samples)

    axes[0, 1].plot(
        x_theory, theoretical_cdf, "r-", linewidth=2, label="Theoretical CDF"
    )
    axes[0, 1].plot(
        x_theory,
        empirical_cdf,
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label="Empirical CDF",
    )
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("P(X ≤ x)")
    axes[0, 1].set_title("CDF Comparison")
    axes[0, 1].legend()

    # Plot 3: Q-Q Plot
    qqplot(
        samples,
        dist=scipy_dist,
        ax=axes[1, 0],
        alpha=0.6,
        line="45",
        markersize=4,
    )
    axes[1, 0].set_title("Q-Q Plot")

    # Plot 4: Sample Realization
    sample_indices = np.arange(min(1000, len(samples)))
    axes[1, 1].plot(
        sample_indices,
        samples[: len(sample_indices)],
        alpha=0.7,
        linewidth=0.8,
    )
    theoretical_mean = scipy_dist.mean()
    axes[1, 1].axhline(
        y=theoretical_mean,
        linestyle="--",
        label=f"Theoretical mean: {theoretical_mean:.3f}",
    )
    sample_mean = np.mean(samples)
    axes[1, 1].axhline(
        y=sample_mean,
        linestyle="--",
        alpha=0.8,
        label=f"Sample mean: {sample_mean:.3f}",
    )
    axes[1, 1].set_xlabel("Sample index")
    axes[1, 1].set_ylabel("Sample value")
    axes[1, 1].set_title("Sample Realization")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def importance_sampling_diagnostic_plots(
    diagnostics: Dict, title: str = "Importance Sampling Diagnostics"
):
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
