import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.mc_lab.independent_metropolis_hastings import independent_metropolis_hastings


# Example: Sample from a standard normal using a wider normal as proposal
def example_normal_target():
    """Example sampling from N(0,1) using N(0,4) as proposal"""

    # Target: Standard normal N(0,1)
    target = stats.norm(0, 1)

    def target_log_pdf(x):
        return target.logpdf(x)

    # Proposal: Wider normal N(0,2)
    proposal = stats.norm(0, 2)

    def proposal_sampler():
        return proposal.rvs()

    def proposal_log_pdf(x):
        return proposal.logpdf(x)

    # Run sampler
    n_samples = 10000
    samples, n_accepted = independent_metropolis_hastings(
        target_log_pdf, proposal_sampler, proposal_log_pdf, n_samples, initial_value=0.0
    )

    print(f"Acceptance rate: {n_accepted / n_samples:.2%}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Trace plot
    axes[0].plot(samples[:1000], alpha=0.7)
    axes[0].set_title("Trace Plot (first 1000 samples)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Value")

    # Histogram
    axes[1].hist(samples[1000:], bins=50, density=True, alpha=0.7, label="Samples")
    x = np.linspace(-4, 4, 100)
    axes[1].plot(x, target.pdf(x), "r-", label="Target N(0,1)")
    axes[1].set_title("Sample Distribution")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # Autocorrelation
    from statsmodels.tsa.stattools import acf

    lags = 50
    autocorr = acf(samples[1000:], nlags=lags)
    axes[2].stem(range(lags + 1), autocorr, linefmt="b-", markerfmt="bo", basefmt=" ")
    axes[2].set_title("Autocorrelation Function")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("ACF")
    axes[2].axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.show()

    return samples


# Example: Sample from a mixture of Gaussians
def example_mixture():
    """Example sampling from a bimodal distribution"""

    # Target: Mixture of two Gaussians
    def target_log_pdf(x):
        # 0.3 * N(-2, 0.5) + 0.7 * N(2, 0.8)
        p1 = 0.3 * stats.norm.pdf(x, -2, 0.5)
        p2 = 0.7 * stats.norm.pdf(x, 2, 0.8)
        return np.log(p1 + p2)

    # Proposal: Wide normal covering both modes
    proposal = stats.norm(0, 3)

    def proposal_sampler():
        return proposal.rvs()

    def proposal_log_pdf(x):
        return proposal.logpdf(x)

    # Run sampler
    n_samples = 20000
    samples, n_accepted = independent_metropolis_hastings(
        target_log_pdf, proposal_sampler, proposal_log_pdf, n_samples, initial_value=0.0
    )

    print(f"Acceptance rate: {n_accepted / n_samples:.2%}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.hist(samples[2000:], bins=100, density=True, alpha=0.7, label="Samples")

    # Plot true density
    x = np.linspace(-6, 6, 1000)
    true_density = 0.3 * stats.norm.pdf(x, -2, 0.5) + 0.7 * stats.norm.pdf(x, 2, 0.8)
    plt.plot(x, true_density, "r-", linewidth=2, label="True density")

    plt.title("Sampling from Bimodal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    return samples


if __name__ == "__main__":
    print("Example 1: Sampling from standard normal")
    samples1 = example_normal_target()

    print("\nExample 2: Sampling from bimodal distribution")
    samples2 = example_mixture()
