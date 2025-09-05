# Reusable visualization function for all distribution examples
import matplotlib.pyplot as plt
import numpy as np


def plot_distribution_analysis(
    samples,
    distribution_name,
    example_number,
    method_description,
    theoretical_mean,
    sample_mean,
    scipy_dist,
    color="blue",
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
    theoretical_mean : float
        Theoretical mean of the distribution
    sample_mean : float
        Sample mean of the generated samples
    scipy_dist : scipy.stats distribution object
        Scipy distribution for comparison (with parameters set)
    color : str, optional
        Color for the plots (default: 'blue')
    x_range : tuple, optional
        (min, max) range for x-axis. If None, will be inferred from samples
    n_points : int, optional
        Number of points for theoretical curves (default: 1000)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Example {example_number}: {distribution_name} Distribution via {method_description}",
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
        color=color,
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
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: CDF Comparison
    theoretical_cdf = scipy_dist.cdf(x_theory)
    empirical_cdf = np.searchsorted(np.sort(samples), x_theory) / len(samples)

    axes[0, 1].plot(
        x_theory, theoretical_cdf, "r-", linewidth=2, label="Theoretical CDF"
    )
    axes[0, 1].plot(
        x_theory,
        empirical_cdf,
        color=color,
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label="Empirical CDF",
    )
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("P(X ≤ x)")
    axes[0, 1].set_title("CDF Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Q-Q Plot
    quantiles = np.linspace(0.01, 0.99, 100)
    theoretical_quantiles = scipy_dist.ppf(quantiles)
    empirical_quantiles = np.quantile(samples, quantiles)

    axes[1, 0].scatter(
        theoretical_quantiles, empirical_quantiles, alpha=0.6, s=20, color=color
    )
    min_val = np.min(theoretical_quantiles)
    max_val = np.max(theoretical_quantiles)
    axes[1, 0].plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect fit",
    )
    axes[1, 0].set_xlabel("Theoretical quantiles")
    axes[1, 0].set_ylabel("Sample quantiles")
    axes[1, 0].set_title("Q-Q Plot")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Sample Realization
    sample_indices = np.arange(min(1000, len(samples)))
    axes[1, 1].plot(
        sample_indices,
        samples[: len(sample_indices)],
        color=color,
        alpha=0.7,
        linewidth=0.8,
    )
    axes[1, 1].axhline(
        y=theoretical_mean,
        color="r",
        linestyle="--",
        label=f"Theoretical mean: {theoretical_mean:.3f}",
    )
    axes[1, 1].axhline(
        y=sample_mean,
        color="orange",
        linestyle="--",
        alpha=0.8,
        label=f"Sample mean: {sample_mean:.3f}",
    )
    axes[1, 1].set_xlabel("Sample index")
    axes[1, 1].set_ylabel("Sample value")
    axes[1, 1].set_title("Sample Realization")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
