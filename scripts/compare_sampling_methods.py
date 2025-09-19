import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gamma, norm

from mc_lab.gibbs_sampler_2d import GibbsSampler2D
from mc_lab.MALA_auto_grad import MALAAutoGradSampler
from mc_lab.metropolis_hastings import MetropolisHastingsSampler


def target_density(x, y):
    """Target probability density (unnormalized): x^2 * exp(-x*y^2 - y^2 + 2*y - 4*x)"""
    return x**2 * np.exp(-x * y**2 - y**2 + 2 * y - 4 * x)


def log_target_density(x, y):
    """Log of target probability density"""
    if x <= 0:  # x must be positive for x^2 term
        return -np.inf
    return 2 * np.log(x) - x * y**2 - y**2 + 2 * y - 4 * x


def log_target_2d(xy):
    """2D log target for Metropolis-Hastings sampler"""
    x, y = xy[0], xy[1]
    return log_target_density(x, y)


def grad_log_target_2d(xy):
    """Gradient of log target density for MALA sampler"""
    x, y = xy[0], xy[1]
    if x <= 0:
        return np.array([-np.inf, -np.inf])

    # Partial derivatives of log_target_density
    grad_x = 2.0 / x - y**2 - 4
    grad_y = -2 * x * y - 2 * y + 2

    return np.array([grad_x, grad_y])


def sample_x_given_y(y):
    """Sample x|y ~ Gamma(3, scale=1/(y^2 + 4))"""
    scale = 1.0 / (y**2 + 4)
    return gamma.rvs(3, scale=scale)


def sample_y_given_x(x):
    """Sample y|x ~ Normal(1/(1+x), scale=1/sqrt(2*(x+1)))"""
    if x <= -1:  # Avoid numerical issues
        x = -0.99
    mean = 1.0 / (1 + x)
    scale = 1.0 / np.sqrt(2 * (x + 1))
    return norm.rvs(mean, scale)


def run_independent_mh():
    """Run Independent Metropolis-Hastings sampler using the class interface"""
    print("Running Independent Metropolis-Hastings...")

    # Use a bivariate normal proposal centered at (1, 1) with moderate spread
    proposal_mean = np.array([1.0, 1.0])
    proposal_cov = np.array([[0.5, 0.1], [0.1, 0.3]])
    proposal_dist = stats.multivariate_normal(proposal_mean, proposal_cov)

    def target_log_pdf(xy):
        return log_target_2d(xy)

    def proposal_sampler():
        return proposal_dist.rvs()

    def proposal_log_pdf(xy):
        return proposal_dist.logpdf(xy)

    # Use the class-based interface for 2D sampling
    from mc_lab.independent_metropolis_hastings import (
        IndependentMetropolisHastingsSampler,
    )

    sampler = IndependentMetropolisHastingsSampler(
        target_log_pdf=target_log_pdf,
        proposal_sampler=proposal_sampler,
        proposal_log_pdf=proposal_log_pdf,
        var_names=["x", "y"],
    )

    idata = sampler.sample(
        n_samples=10000,
        n_chains=1,
        burn_in=2000,
        initial_states=np.array([1.0, 1.0]),
        progressbar=False,
    )

    # Extract samples
    samples = np.column_stack(
        [idata.posterior["x"].values[0], idata.posterior["y"].values[0]]
    )

    acceptance_rates = sampler.get_acceptance_rates(idata)
    acceptance_rate = acceptance_rates["overall"]
    print(f"Independent MH acceptance rate: {acceptance_rate:.3f}")

    return samples, acceptance_rate


def run_gibbs_sampler():
    """Run Gibbs sampler"""
    print("Running Gibbs sampler...")

    def log_joint(x, y):
        return log_target_density(x, y)

    sampler = GibbsSampler2D(
        sample_x_given_y=sample_x_given_y,
        sample_y_given_x=sample_y_given_x,
        log_target=log_joint,
        var_names=("x", "y"),
    )

    idata = sampler.sample(
        n_samples=10000,
        n_chains=1,
        burn_in=2000,
        initial_states=np.array([1.0, 1.0]),
        progressbar=False,
    )

    # Extract samples
    samples = np.column_stack(
        [idata.posterior["x"].values[0], idata.posterior["y"].values[0]]
    )

    print("Gibbs sampler acceptance rate: 1.000 (always accepts)")

    return samples, 1.0


def run_metropolis_hastings():
    """Run Metropolis-Hastings with Random Walk"""
    print("Running Metropolis-Hastings with Random Walk...")

    sampler = MetropolisHastingsSampler(
        log_target=log_target_2d,
        proposal_scale=np.array([0.2, 0.15]),  # Tuned for reasonable acceptance
        var_names=["x", "y"],
        adaptive_scaling=True,
    )

    idata = sampler.sample(
        n_samples=10000,
        n_chains=1,
        burn_in=2000,
        initial_states=np.array([1.0, 1.0]),
        progressbar=False,
    )

    # Extract samples
    samples = np.column_stack(
        [idata.posterior["x"].values[0], idata.posterior["y"].values[0]]
    )

    acceptance_rates = sampler.get_acceptance_rates(idata)
    acceptance_rate = acceptance_rates["overall"]
    print(f"Metropolis-Hastings acceptance rate: {acceptance_rate:.3f}")

    return samples, acceptance_rate


def run_mala_sampler():
    """Run MALA (Metropolis-adjusted Langevin Algorithm) with automatic gradients"""
    print("Running MALA sampler with automatic gradients...")

    # Import torch for PyTorch-compatible log target
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for MALAAutoGradSampler")

    def log_target_torch(xy):
        """PyTorch-compatible log target function"""
        x, y = xy[0], xy[1]
        # Use torch.where to handle x <= 0 case in a differentiable way
        # Add small epsilon to avoid log(0) and ensure x > 0
        x_safe = torch.where(x > 0, x, torch.tensor(1e-10, dtype=x.dtype))
        log_prob = 2 * torch.log(x_safe) - x_safe * y**2 - y**2 + 2 * y - 4 * x_safe
        # Return very negative value for x <= 0
        return torch.where(x > 0, log_prob, torch.tensor(-1e10, dtype=x.dtype))

    sampler = MALAAutoGradSampler(
        log_target=log_target_torch,
        step_size=0.15,  # Tuned for reasonable acceptance and good mixing
        var_names=["x", "y"],
    )

    idata = sampler.sample(
        n_samples=10000,
        n_chains=1,
        burn_in=2000,  # Longer burn-in for better convergence
        initial_states=np.array([0.5, 0.5]),  # Start closer to the mode
        progressbar=False,
    )

    # Extract samples
    samples = np.column_stack(
        [idata.posterior["x"].values[0], idata.posterior["y"].values[0]]
    )

    acceptance_rates = sampler.get_acceptance_rates(idata)
    acceptance_rate = acceptance_rates["overall"]
    print(f"MALA acceptance rate: {acceptance_rate:.3f}")

    return samples, acceptance_rate


def compare_methods():
    """Compare all four sampling methods"""

    # Run all samplers
    samples_imh, acc_imh = run_independent_mh()
    samples_gibbs, acc_gibbs = run_gibbs_sampler()
    samples_mh, acc_mh = run_metropolis_hastings()
    samples_mala, acc_mala = run_mala_sampler()

    # Create comparison plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    methods = [
        ("Independent MH", samples_imh, acc_imh),
        ("Gibbs Sampler", samples_gibbs, acc_gibbs),
        ("Metropolis-Hastings", samples_mh, acc_mh),
        ("MALA", samples_mala, acc_mala),
    ]

    colors = ["#332288", "#117733", "#44AA99", "#88CCEE"]

    # Determine common axis limits for marginals
    all_x_samples = np.concatenate([samples[:, 0] for _, samples, _ in methods])
    all_y_samples = np.concatenate([samples[:, 1] for _, samples, _ in methods])

    # Filter out negative x values for consistent comparison
    valid_x = all_x_samples[all_x_samples > 0]

    x_range = (np.min(valid_x), np.max(valid_x))
    y_range = (np.min(all_y_samples), np.max(all_y_samples))

    # Create common bins for consistent histograms
    x_bins = np.linspace(x_range[0], x_range[1], 50)
    y_bins = np.linspace(y_range[0], y_range[1], 50)

    # Plot marginal distributions and collect max densities for y-axis alignment
    x_max_density = 0
    y_max_density = 0

    # First pass: plot histograms and find max densities
    for i, (name, samples, acc) in enumerate(methods):
        # Filter x samples to remove negatives
        valid_x_mask = samples[:, 0] > 0
        valid_x_samples = samples[valid_x_mask, 0]

        # X marginal
        n_x, _, _ = axes[0, i].hist(
            valid_x_samples,
            bins=x_bins,
            density=True,
            alpha=0.7,
            color=colors[i],
            label=f"{name}\nAcc: {acc:.3f}",
        )
        x_max_density = max(x_max_density, np.max(n_x))

        axes[0, i].set_title(f"{name}: X Marginal")
        axes[0, i].set_xlabel("x")
        axes[0, i].set_ylabel("Density")
        axes[0, i].set_xlim(x_range)
        axes[0, i].legend()

        # Y marginal
        n_y, _, _ = axes[1, i].hist(
            samples[:, 1],
            bins=y_bins,
            density=True,
            alpha=0.7,
            color=colors[i],
            label=f"{name}\nAcc: {acc:.3f}",
        )
        y_max_density = max(y_max_density, np.max(n_y))

        axes[1, i].set_title(f"{name}: Y Marginal")
        axes[1, i].set_xlabel("y")
        axes[1, i].set_ylabel("Density")
        axes[1, i].set_xlim(y_range)
        axes[1, i].legend()

    # Second pass: set common y-axis limits
    for i in range(len(methods)):
        axes[0, i].set_ylim(0, x_max_density * 1.05)
        axes[1, i].set_ylim(0, y_max_density * 1.05)

    plt.tight_layout()
    plt.show()

    # Joint distribution comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Collect all valid samples to determine common axis limits
    all_valid_samples = []
    for name, samples, acc in methods:
        valid_mask = samples[:, 0] > 0
        valid_samples = samples[valid_mask]
        all_valid_samples.append(valid_samples)

    # Combine all samples to find common limits
    combined_samples = np.vstack(all_valid_samples)
    x_min, x_max = np.min(combined_samples[:, 0]), np.max(combined_samples[:, 0])
    y_min, y_max = np.min(combined_samples[:, 1]), np.max(combined_samples[:, 1])

    # Add small margin
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    x_lims = (x_min - x_margin, x_max + x_margin)
    y_lims = (y_min - y_margin, y_max + y_margin)

    for i, (name, samples, acc) in enumerate(methods):
        # Remove any negative x values for visualization
        valid_mask = samples[:, 0] > 0
        valid_samples = samples[valid_mask]

        axes[i].scatter(
            valid_samples[:, 0], valid_samples[:, 1], alpha=0.5, s=1, color=colors[i]
        )
        axes[i].set_title(f"{name}\nAcceptance: {acc:.3f}")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].grid(True, alpha=0.3)

        # Set common axis limits
        axes[i].set_xlim(x_lims)
        axes[i].set_ylim(y_lims)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for name, samples, acc in methods:
        valid_mask = samples[:, 0] > 0
        valid_samples = samples[valid_mask]

        print(f"\n{name}:")
        print(f"  Acceptance rate: {acc:.3f}")
        print(
            f"  Valid samples: {np.sum(valid_mask)}/{len(samples)} ({100 * np.sum(valid_mask) / len(samples):.1f}%)"
        )
        print(
            f"  X: mean={np.mean(valid_samples[:, 0]):.3f}, std={np.std(valid_samples[:, 0]):.3f}"
        )
        print(
            f"  Y: mean={np.mean(valid_samples[:, 1]):.3f}, std={np.std(valid_samples[:, 1]):.3f}"
        )
        print(
            f"  Correlation: {np.corrcoef(valid_samples[:, 0], valid_samples[:, 1])[0, 1]:.3f}"
        )


def plot_target_contours():
    """Plot contours of the target distribution for reference"""
    print("\nPlotting target distribution contours...")

    x = np.linspace(0.1, 3, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = target_density(X[j, i], Y[j, i])

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=20, colors="black", alpha=0.5)
    plt.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.7)
    plt.colorbar(label="Density")
    plt.title("Target Distribution: $x^2 e^{-xy^2 - y^2 + 2y - 4x}$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    print("Comparing Four MCMC Sampling Methods")
    print("=" * 50)
    print("Target: x² * exp(-x*y² - y² + 2*y - 4*x)")
    print("=" * 50)

    # Show target distribution
    plot_target_contours()

    # Run comparison
    compare_methods()

    print("\nComparison complete!")
