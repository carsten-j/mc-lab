from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def barker_proposal_1d(
    x: float, grad_log_pi: float, sigma: float, rng: np.random.Generator
) -> float:
    """
    Algorithm 1: Generate a Barker proposal in 1D

    Args:
        x: Current position
        grad_log_pi: Gradient of log π at x
        sigma: Proposal scale parameter
        rng: Random number generator

    Returns:
        Proposed next position y
    """
    # (a) Draw z ~ μ_σ (Gaussian with std σ)
    z = rng.normal(0, sigma)

    # (b) Calculate p(x,z) = 1/(1 + e^(-z∇log π(x)))
    p_xz = 1.0 / (1.0 + np.exp(-z * grad_log_pi))

    # (c) Set b(x,z) = 1 with probability p(x,z), and b(x,z) = -1 otherwise
    u = rng.uniform()
    if u < p_xz:
        b = 1
    else:
        b = -1

    # (d) Set y = x + b(x,z) × z
    y = x + b * z

    return y


def barker_mcmc(
    target_log_density: Callable[[np.ndarray], float],
    target_grad_log_density: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    sigma: float,
    n_iterations: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Algorithm 2: Metropolis-Hastings with the Barker proposal on ℝ^d

    Args:
        target_log_density: Function computing log π(x)
        target_grad_log_density: Function computing ∇log π(x)
        x0: Initial position (d-dimensional)
        sigma: Proposal scale parameter
        n_iterations: Number of MCMC iterations
        seed: Random seed for reproducibility

    Returns:
        samples: Array of samples (n_iterations × d)
        accepted: Binary array indicating accepted proposals
        acceptance_rates: Running acceptance rates
    """
    rng = np.random.default_rng(seed)
    d = len(x0)

    # Initialize storage
    samples = np.zeros((n_iterations, d))
    accepted = np.zeros(n_iterations, dtype=bool)
    acceptance_rates = []

    # Initialize chain
    x = x0.copy()
    samples[0] = x
    log_pi_x = target_log_density(x)

    n_accepted = 0

    for t in range(1, n_iterations):
        # (a) Given x^(t) = x, draw y_i using Algorithm 1 independently for i ∈ {1,...,d}
        grad_log_pi_x = target_grad_log_density(x)
        y = np.zeros(d)

        for i in range(d):
            y[i] = barker_proposal_1d(x[i], grad_log_pi_x[i], sigma, rng)

        # (b) Set x^(t+1) = y with probability α^B(x,y)
        # Compute acceptance probability
        log_pi_y = target_log_density(y)
        grad_log_pi_y = target_grad_log_density(y)

        # Compute the product term in the acceptance probability
        log_ratio_forward = 0.0
        log_ratio_backward = 0.0

        for i in range(d):
            # Forward: from x to y
            z_forward = y[i] - x[i]
            log_ratio_forward += np.log(1 + np.exp(-(x[i] - y[i]) * grad_log_pi_x[i]))

            # Backward: from y to x
            z_backward = x[i] - y[i]
            log_ratio_backward += np.log(1 + np.exp(-(y[i] - x[i]) * grad_log_pi_y[i]))

        # Compute α^B(x,y) = min(1, π(y)/π(x) × ∏_i (1+e^(-(x_i-y_i)∂_i log π(x)))/(1+e^(-(y_i-x_i)∂_i log π(y))))
        log_alpha = log_pi_y - log_pi_x + log_ratio_forward - log_ratio_backward
        alpha = min(1.0, np.exp(log_alpha))

        # Accept or reject
        u = rng.uniform()
        if u < alpha:
            x = y.copy()
            log_pi_x = log_pi_y
            accepted[t] = True
            n_accepted += 1
        else:
            accepted[t] = False

        samples[t] = x
        acceptance_rates.append(n_accepted / t)

    return samples, accepted, acceptance_rates


# Example usage for different target distributions
def example_1d():
    """Example: 1D Gaussian target"""
    # Define target: Standard normal
    target_log_density = lambda x: -0.5 * x[0] ** 2
    target_grad_log_density = lambda x: np.array([-x[0]])

    # Run MCMC
    x0 = np.array([2.0])
    samples, accepted, acc_rates = barker_mcmc(
        target_log_density, target_grad_log_density, x0, sigma=1.0, n_iterations=10000
    )

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Trace plot
    axes[0].plot(samples[:1000, 0], alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("x")
    axes[0].set_title("Trace Plot (first 1000 iterations)")
    axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(samples[1000:, 0], bins=50, density=True, alpha=0.7, edgecolor="black")
    x_range = np.linspace(-4, 4, 100)
    axes[1].plot(x_range, norm.pdf(x_range), "r-", label="True density")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Marginal Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Acceptance rate
    axes[2].plot(acc_rates)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Acceptance Rate")
    axes[2].set_title(f"Running Acceptance Rate (Final: {acc_rates[-1]:.3f})")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return samples


def example_2d():
    """Example: 2D Gaussian target with correlation"""
    # Define target: 2D Gaussian with correlation
    # Covariance matrix
    rho = 0.8
    Sigma = np.array([[1, rho], [rho, 1]])
    Sigma_inv = np.linalg.inv(Sigma)

    target_log_density = lambda x: -0.5 * x @ Sigma_inv @ x
    target_grad_log_density = lambda x: -Sigma_inv @ x

    # Run MCMC
    x0 = np.array([2.0, 2.0])
    samples, accepted, acc_rates = barker_mcmc(
        target_log_density, target_grad_log_density, x0, sigma=0.8, n_iterations=10000
    )

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Trace plots
    for i in range(2):
        axes[i, 0].plot(samples[:1000, i], alpha=0.7)
        axes[i, 0].set_xlabel("Iteration")
        axes[i, 0].set_ylabel(f"x_{i + 1}")
        axes[i, 0].set_title(f"Trace Plot x_{i + 1} (first 1000 iterations)")
        axes[i, 0].grid(True, alpha=0.3)

    # Joint distribution
    axes[0, 1].scatter(samples[1000:, 0], samples[1000:, 1], alpha=0.3, s=1)
    axes[0, 1].set_xlabel("x_1")
    axes[0, 1].set_ylabel("x_2")
    axes[0, 1].set_title("Joint Distribution")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect("equal")

    # Add contours of true distribution
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    from scipy.stats import multivariate_normal

    rv = multivariate_normal([0, 0], Sigma)
    axes[0, 1].contour(X, Y, rv.pdf(pos), colors="red", alpha=0.5, levels=5)

    # Marginal histograms
    axes[1, 1].hist(
        samples[1000:, 0], bins=50, density=True, alpha=0.7, edgecolor="black"
    )
    x_range = np.linspace(-4, 4, 100)
    axes[1, 1].plot(x_range, norm.pdf(x_range), "r-", label="True marginal")
    axes[1, 1].set_xlabel("x_1")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Marginal x_1")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Acceptance rate
    axes[0, 2].plot(acc_rates)
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("Acceptance Rate")
    axes[0, 2].set_title(f"Running Acceptance Rate (Final: {acc_rates[-1]:.3f})")
    axes[0, 2].grid(True, alpha=0.3)

    # Autocorrelation function for x_1
    from statsmodels.tsa.stattools import acf

    lags = 50
    acf_vals = acf(samples[1000:, 0], nlags=lags)
    axes[1, 2].bar(range(lags + 1), acf_vals, alpha=0.7)
    axes[1, 2].set_xlabel("Lag")
    axes[1, 2].set_ylabel("ACF")
    axes[1, 2].set_title("Autocorrelation Function (x_1)")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return samples


# Run examples
if __name__ == "__main__":
    print("Running 1D example...")
    samples_1d = example_1d()
    print(
        f"1D example complete. Sample mean: {np.mean(samples_1d[1000:]):.3f}, "
        f"Sample std: {np.std(samples_1d[1000:]):.3f}"
    )

    print("\nRunning 2D example...")
    samples_2d = example_2d()
    print(f"2D example complete. Sample means: {np.mean(samples_2d[1000:], axis=0)}")
    print(f"Sample covariance:\n{np.cov(samples_2d[1000:].T)}")
