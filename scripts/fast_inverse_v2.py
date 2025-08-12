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


# Example and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example 1: Gaussian distribution
    print("Testing Gaussian distribution...")
    gaussian_sampler = create_gaussian_sampler(mean=2, std=1.5)
    gaussian_samples = gaussian_sampler.sample(1000)

    # Example 2: Custom distribution (mixture of Gaussians)
    print("Testing custom mixture distribution...")

    def mixture_pdf(x):
        return 0.3 * np.exp(-0.5 * ((x + 2) / 0.5) ** 2) / (
            0.5 * np.sqrt(2 * np.pi)
        ) + 0.7 * np.exp(-0.5 * ((x - 1) / 1.0) ** 2) / (1.0 * np.sqrt(2 * np.pi))

    mixture_sampler = ChebyshevSampler(mixture_pdf, [-5, 5])
    mixture_samples = mixture_sampler.sample(1000)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Gaussian samples
    ax1.hist(gaussian_samples, bins=50, density=True, alpha=0.7, label="Samples")
    x_gauss = np.linspace(-3, 7, 1000)
    y_gauss = (1 / (1.5 * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_gauss - 2) / 1.5) ** 2
    )
    ax1.plot(x_gauss, y_gauss, "r-", linewidth=2, label="True PDF")
    ax1.set_title("Gaussian Distribution (μ=2, σ=1.5)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot mixture samples
    ax2.hist(mixture_samples, bins=50, density=True, alpha=0.7, label="Samples")
    x_mix = np.linspace(-5, 5, 1000)
    y_mix = mixture_pdf(x_mix)
    ax2.plot(x_mix, y_mix, "r-", linewidth=2, label="True PDF")
    ax2.set_title("Mixture of Gaussians")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(
        f"Gaussian samples: mean={np.mean(gaussian_samples):.3f}, std={np.std(gaussian_samples):.3f}"
    )
    print(
        f"Mixture samples: mean={np.mean(mixture_samples):.3f}, std={np.std(mixture_samples):.3f}"
    )
