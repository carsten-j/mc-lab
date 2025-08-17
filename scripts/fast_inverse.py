import numpy as np

from mc_lab.fast_inverse_transform import ChebyshevSampler


# Example usage and convenience functions
def create_gaussian_sampler(mean=0, std=1, domain=None):
    """
    Convenience function to create a Gaussian sampler.

    Parameters:
    -----------
    mean : float
        Mean of the Gaussian distribution
    std : float
        Standard deviation of the Gaussian distribution
    domain : list or None
        Domain for sampling. If None, uses mean ± 5*std

    Returns:
    --------
    ChebyshevSampler
        Configured sampler for Gaussian distribution
    """
    if domain is None:
        domain = [mean - 5 * std, mean + 5 * std]

    def gaussian_pdf(x):
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    return ChebyshevSampler(gaussian_pdf, domain)


def create_beta_sampler(alpha, beta, domain=[0, 1]):
    """
    Convenience function to create a Beta distribution sampler.

    Parameters:
    -----------
    alpha, beta : float
        Shape parameters of the Beta distribution
    domain : list
        Domain for the Beta distribution (default [0, 1])

    Returns:
    --------
    ChebyshevSampler
        Configured sampler for Beta distribution
    """
    from scipy.special import gamma

    def beta_pdf(x):
        if np.any((x <= 0) | (x >= 1)):
            return np.where(
                (x <= 0) | (x >= 1),
                0,
                (x ** (alpha - 1) * (1 - x) ** (beta - 1))
                / (gamma(alpha) * gamma(beta) / gamma(alpha + beta)),
            )
        return (x ** (alpha - 1) * (1 - x) ** (beta - 1)) / (
            gamma(alpha) * gamma(beta) / gamma(alpha + beta)
        )

    return ChebyshevSampler(beta_pdf, domain)


def create_multimodal_sampler(domain=[-8, 8]):
    """
    Convenience function to create the multimodal density sampler from the paper.

    PDF: exp(-x²/2) * (1 + sin²(3x)) * (1 + cos²(5x))

    Parameters:
    -----------
    domain : list
        Domain for sampling (default [-8, 8] as in paper)

    Returns:
    --------
    ChebyshevSampler
        Configured sampler for multimodal distribution
    """

    def multimodal_pdf(x):
        return np.exp(-(x**2) / 2) * (1 + np.sin(3 * x) ** 2) * (1 + np.cos(5 * x) ** 2)

    return ChebyshevSampler(multimodal_pdf, domain)


def create_gue_spectral_sampler(domain=[-4, 4]):
    """
    Convenience function to create the 4×4 Gaussian Unitary Ensemble spectral density sampler.

    PDF: exp(-4x²) * (9 + 72x² - 192x⁴ + 512x⁶)

    Parameters:
    -----------
    domain : list
        Domain for sampling (default [-4, 4] as in paper)

    Returns:
    --------
    ChebyshevSampler
        Configured sampler for GUE spectral distribution
    """

    def gue_spectral_pdf(x):
        return np.exp(-4 * x**2) * (9 + 72 * x**2 - 192 * x**4 + 512 * x**6)

    return ChebyshevSampler(gue_spectral_pdf, domain)


def create_oscillatory_sampler(domain=[-1, 1]):
    """
    Convenience function to create the compactly supported oscillatory density sampler.

    PDF: 2 + cos(100x)

    Parameters:
    -----------
    domain : list
        Domain for sampling (default [-1, 1] as in paper)

    Returns:
    --------
    ChebyshevSampler
        Configured sampler for oscillatory distribution
    """

    def oscillatory_pdf(x):
        return 2 + np.cos(100 * x)

    return ChebyshevSampler(oscillatory_pdf, domain)


def create_concentrated_sech_sampler(domain=[-1, 1]):
    """
    Convenience function to create the concentrated sech density sampler.

    PDF: sech(200x) = 2/(exp(200x) + exp(-200x))

    Parameters:
    -----------
    domain : list
        Domain for sampling (default [-1, 1] as in paper)

    Returns:
    --------
    ChebyshevSampler
        Configured sampler for concentrated sech distribution
    """

    def concentrated_sech_pdf(x):
        # sech(x) = 2/(exp(x) + exp(-x)) = 1/cosh(x)
        return 1.0 / np.cosh(200 * x)

    return ChebyshevSampler(concentrated_sech_pdf, domain)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test all distributions from the paper
    print("Testing distributions from 'Fast inverse transform sampling' paper...")

    # Create samplers for all distributions
    samplers = {
        "Gaussian": create_gaussian_sampler(mean=2, std=1.5),
        "Multimodal": create_multimodal_sampler(),
        "GUE Spectral": create_gue_spectral_sampler(),
        "Oscillatory": create_oscillatory_sampler(),
        "Concentrated Sech": create_concentrated_sech_sampler(),
    }

    # Generate samples
    N = 1000
    samples = {}
    for name, sampler in samplers.items():
        print(f"Sampling from {name} distribution...")
        samples[name] = sampler.sample(N)

    # Custom mixture for comparison
    print("Testing custom mixture distribution...")

    def mixture_pdf(x):
        return 0.3 * np.exp(-0.5 * ((x + 2) / 0.5) ** 2) / (
            0.5 * np.sqrt(2 * np.pi)
        ) + 0.7 * np.exp(-0.5 * ((x - 1) / 1.0) ** 2) / (1.0 * np.sqrt(2 * np.pi))

    mixture_sampler = ChebyshevSampler(mixture_pdf, [-5, 5])
    samples["Custom Mixture"] = mixture_sampler.sample(N)

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Define true PDFs and domains for plotting
    plot_configs = {
        "Gaussian": {
            "domain": np.linspace(-3, 7, 1000),
            "pdf": lambda x: (1 / (1.5 * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * ((x - 2) / 1.5) ** 2),
        },
        "Multimodal": {
            "domain": np.linspace(-8, 8, 1000),
            "pdf": lambda x: np.exp(-(x**2) / 2)
            * (1 + np.sin(3 * x) ** 2)
            * (1 + np.cos(5 * x) ** 2),
        },
        "GUE Spectral": {
            "domain": np.linspace(-4, 4, 1000),
            "pdf": lambda x: np.exp(-4 * x**2)
            * (9 + 72 * x**2 - 192 * x**4 + 512 * x**6),
        },
        "Oscillatory": {
            "domain": np.linspace(-1, 1, 1000),
            "pdf": lambda x: 2 + np.cos(100 * x),
        },
        "Concentrated Sech": {
            "domain": np.linspace(-1, 1, 1000),
            "pdf": lambda x: 1.0 / np.cosh(200 * x),
        },
        "Custom Mixture": {"domain": np.linspace(-5, 5, 1000), "pdf": mixture_pdf},
    }

    for i, (name, sample_data) in enumerate(samples.items()):
        ax = axes[i]
        config = plot_configs[name]

        # Plot histogram of samples
        ax.hist(
            sample_data,
            bins=50,
            density=True,
            alpha=0.7,
            label="Samples",
            color="skyblue",
        )

        # Plot true PDF
        x_true = config["domain"]
        y_true = config["pdf"](x_true)
        # Normalize for visualization
        y_true = y_true / np.trapz(y_true, x_true)
        ax.plot(x_true, y_true, "r-", linewidth=2, label="True PDF")

        ax.set_title(f"{name} Distribution")
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Print statistics
        print(f"{name}: mean={np.mean(sample_data):.3f}, std={np.std(sample_data):.3f}")

    plt.tight_layout()
    plt.show()

    # Performance comparison example
    print("\nPerformance test: sampling efficiency...")
    multimodal_sampler = create_multimodal_sampler()

    # First call (includes setup time)
    import time

    start = time.time()
    first_samples = multimodal_sampler.sample(1000)
    first_time = time.time() - start

    # Second call (uses cache)
    start = time.time()
    second_samples = multimodal_sampler.sample(1000)
    second_time = time.time() - start

    print(f"First sampling (with setup): {first_time:.4f} seconds")
    print(f"Second sampling (cached): {second_time:.4f} seconds")
    print(f"Speedup factor: {first_time / second_time:.1f}x")
