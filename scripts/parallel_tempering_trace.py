"""Random Walk Metropolis MCMC for tempered bimodal Gaussian distributions.

This script implements a standard RWM MCMC algorithm to sample from multiple
chains at different temperatures (gamma values) without any swapping between chains.
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from mc_lab._rng import RandomState, as_generator


def bimodal_gaussian_log_pdf(
    x: float, mu1: float = -2.0, mu2: float = 2.0, sigma: float = 0.5
) -> float:
    """Compute log of bimodal Gaussian PDF.

    Args:
        x: Point at which to evaluate the log PDF
        mu1: Mean of first Gaussian
        mu2: Mean of second Gaussian
        sigma: Standard deviation (same for both)

    Returns:
        Log PDF value at x
    """
    log_norm = -0.5 * np.log(2 * np.pi) - np.log(sigma)
    log_pdf1 = log_norm - 0.5 * ((x - mu1) / sigma) ** 2
    log_pdf2 = log_norm - 0.5 * ((x - mu2) / sigma) ** 2

    # log(0.5 * (exp(a) + exp(b))) = log(0.5) + log(exp(a) + exp(b))
    # Use log-sum-exp trick for numerical stability
    max_log = max(log_pdf1, log_pdf2)
    return (
        np.log(0.5)
        + max_log
        + np.log(np.exp(log_pdf1 - max_log) + np.exp(log_pdf2 - max_log))
    )


def tempered_log_pdf(log_pdf_func: Callable, x: float, gamma: float) -> float:
    """Apply tempering to a log PDF.

    Args:
        log_pdf_func: Function computing log PDF
        x: Point at which to evaluate
        gamma: Tempering parameter (0 < gamma <= 1)

    Returns:
        Tempered log PDF value: gamma * log_pdf(x)
    """
    return gamma * log_pdf_func(x)


def rwm_mcmc(
    log_pdf_func: Callable[[float], float],
    x0: float,
    n_samples: int,
    proposal_std: float = 0.5,
    rng: RandomState = None,
) -> tuple[np.ndarray, float]:
    """Run Random Walk Metropolis MCMC sampler.

    Args:
        log_pdf_func: Function computing log PDF of target distribution
        x0: Initial state
        n_samples: Number of samples to generate
        proposal_std: Standard deviation of Gaussian proposal
        rng: Random number generator

    Returns:
        Tuple of (samples array, acceptance rate)
    """
    rng = as_generator(rng)

    samples = np.zeros(n_samples)
    samples[0] = x0

    x_current = x0
    log_pdf_current = log_pdf_func(x_current)
    n_accepted = 0

    for i in range(1, n_samples):
        # Propose new state
        x_proposed = x_current + rng.normal(0, proposal_std)

        # Compute acceptance probability
        log_pdf_proposed = log_pdf_func(x_proposed)
        log_alpha = log_pdf_proposed - log_pdf_current

        # Accept or reject
        if np.log(rng.random()) < log_alpha:
            x_current = x_proposed
            log_pdf_current = log_pdf_proposed
            n_accepted += 1

        samples[i] = x_current

    acceptance_rate = n_accepted / (n_samples - 1)
    return samples, acceptance_rate


def run_parallel_chains(
    n_chains: int = 10,
    n_samples: int = 10000,
    gamma_min: float = 0.1,
    gamma_max: float = 1.0,
    proposal_std: float = 0.5,
    x0: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run multiple MCMC chains at different temperatures.

    Args:
        n_chains: Number of parallel chains (one per temperature)
        n_samples: Number of samples per chain
        gamma_min: Minimum temperature parameter
        gamma_max: Maximum temperature parameter
        proposal_std: Standard deviation for RWM proposals
        x0: Initial state for all chains
        seed: Random seed

    Returns:
        Tuple of (samples array [n_chains, n_samples], gamma values, acceptance rates)
    """
    # Create temperature ladder
    gammas = np.linspace(gamma_min, gamma_max, n_chains)

    # Storage for all chains
    all_samples = np.zeros((n_chains, n_samples))
    acceptance_rates = np.zeros(n_chains)

    # Run each chain independently
    rng = as_generator(seed)
    for i, gamma in enumerate(gammas):
        print(f"Running chain {i + 1}/{n_chains} with γ={gamma:.2f}...")

        # Create tempered log PDF for this chain
        def tempered_log_pdf_i(x: float) -> float:
            return tempered_log_pdf(bimodal_gaussian_log_pdf, x, gamma)

        # Run MCMC
        samples, acc_rate = rwm_mcmc(
            tempered_log_pdf_i, x0, n_samples, proposal_std, rng=rng
        )

        all_samples[i] = samples
        acceptance_rates[i] = acc_rate

        print(f"  Acceptance rate: {acc_rate:.3f}")

    return all_samples, gammas, acceptance_rates


def plot_trace(
    samples: np.ndarray,
    gammas: np.ndarray,
    show_chains: tuple[int, ...] = (7, 8, 9),
    colors: tuple[str, ...] = ("#117733", "#CC6677", "#332288"),
) -> None:
    """Create trace plot for selected chains.

    Args:
        samples: Sample array of shape [n_chains, n_samples]
        gammas: Temperature values for each chain
        show_chains: Indices of chains to display (0-indexed)
        colors: Colors for each displayed chain
    """
    n_chains, n_samples = samples.shape

    plt.figure(figsize=(12, 6))

    # Plot selected chains
    for idx, chain_idx in enumerate(show_chains):
        if chain_idx < n_chains:
            color = colors[idx % len(colors)]
            plt.plot(
                samples[chain_idx],
                color=color,
                alpha=0.7,
                linewidth=0.5,
                label=f"chain {chain_idx + 1}",
            )

    plt.xlabel("iteration", fontsize=12)
    plt.ylabel("X", fontsize=12)
    plt.title(
        f"Let's use N = {n_chains} chains and "
        f"γ₁ = {gammas[0]:.1f}, γ₂ = {gammas[1]:.1f}, ..., γ₁₀ = {gammas[-1]:.1f}. "
        "No swapping.",
        fontsize=12,
    )

    # Create legend with custom labels
    legend_labels = [f"# chain {i + 1}" for i in show_chains]
    plt.legend(legend_labels, fontsize=10, loc="upper right")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    output_path = "parallel_tempering_trace.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nTrace plot saved to: {output_path}")

    plt.show()


def main() -> None:
    """Main function to run the simulation and create plots."""
    print("Running Parallel Tempered MCMC (no swapping)...\n")

    # Run parallel chains
    samples, gammas, acc_rates = run_parallel_chains(
        n_chains=10,
        n_samples=10000,
        gamma_min=0.1,
        gamma_max=1.0,
        proposal_std=0.5,
        x0=0.0,
        seed=42,
    )

    print(
        f"\nOverall acceptance rate range: [{acc_rates.min():.3f}, {acc_rates.max():.3f}]"
    )

    # Create trace plot for chains 8, 9, 10 (indices 7, 8, 9)
    plot_trace(samples, gammas, show_chains=(7, 8, 9))


if __name__ == "__main__":
    main()
