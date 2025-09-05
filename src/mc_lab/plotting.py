# Reusable visualization function for all distribution examples
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
