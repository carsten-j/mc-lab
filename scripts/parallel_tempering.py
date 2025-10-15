import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy import stats
from scipy.integrate import trapezoid


class ParallelTemperingDoubleWell:
    """
    Parallel Tempering for double-well potential U(x) = gamma * (x^2 - 1)^2
    """

    def __init__(self, gamma=4.0, n_chains=5, max_temp=10.0):
        """
        Initialize Parallel Tempering sampler for double-well potential

        Parameters:
        -----------
        gamma : float
            Parameter controlling the height of the barrier between wells
        n_chains : int
            Number of parallel chains
        max_temp : float
            Maximum temperature for the hottest chain
        """
        self.gamma = gamma
        self.n_chains = n_chains

        # Create temperature ladder (geometric spacing)
        self.temps = np.geomspace(1.0, max_temp, n_chains)

    def U(self, x):
        """Potential function U(x) = gamma * (x^2 - 1)^2"""
        return self.gamma * (x**2 - 1) ** 2

    def log_target(self, x, temp=1.0):
        """Log of tempered target distribution"""
        return -self.U(x) / temp

    def metropolis_step(self, x_current, temp, step_size=0.5):
        """Single Metropolis-Hastings step"""
        # Propose new state
        x_proposed = x_current + np.random.normal(0, step_size)

        # Compute acceptance probability
        log_alpha = self.log_target(x_proposed, temp) - self.log_target(x_current, temp)

        # Accept or reject
        if np.log(np.random.rand()) < log_alpha:
            return x_proposed, True
        return x_current, False

    def attempt_swap(self, x_i, x_j, temp_i, temp_j):
        """Attempt to swap states between two chains"""
        # Compute swap acceptance probability
        log_alpha = (self.log_target(x_j, temp_i) + self.log_target(x_i, temp_j)) - (
            self.log_target(x_i, temp_i) + self.log_target(x_j, temp_j)
        )

        # Accept or reject swap
        if np.log(np.random.rand()) < log_alpha:
            return x_j, x_i, True
        return x_i, x_j, False

    def run(
        self,
        n_iterations=10000,
        step_size=0.5,
        swap_interval=10,
        initial_x=None,
        verbose=True,
    ):
        """
        Run parallel tempering simulation

        Returns:
        --------
        chains : array
            Samples from all chains
        swaps : list
            Record of successful swaps
        accept_rates : array
            Acceptance rates for each chain
        """
        # Initialize chains
        if initial_x is None:
            chains_current = np.random.randn(self.n_chains)
        else:
            chains_current = np.array([initial_x] * self.n_chains)

        # Storage
        chains_history = np.zeros((n_iterations, self.n_chains))
        accept_counts = np.zeros(self.n_chains)
        swap_accepts = []
        swap_attempts = 0

        # Main MCMC loop
        for it in range(n_iterations):
            # Update each chain with Metropolis step
            for i in range(self.n_chains):
                chains_current[i], accepted = self.metropolis_step(
                    chains_current[i], self.temps[i], step_size
                )
                if accepted:
                    accept_counts[i] += 1

            # Attempt chain swaps
            if it % swap_interval == 0 and it > 0:
                # Choose random adjacent pair
                i = np.random.randint(0, self.n_chains - 1)
                j = i + 1

                # Attempt swap
                chains_current[i], chains_current[j], swapped = self.attempt_swap(
                    chains_current[i], chains_current[j], self.temps[i], self.temps[j]
                )

                swap_attempts += 1
                if swapped:
                    swap_accepts.append((it, i, j))

            # Store current state
            chains_history[it] = chains_current.copy()

            # Progress update
            if verbose and (it + 1) % (n_iterations // 10) == 0:
                swap_rate = (
                    len(swap_accepts) / swap_attempts if swap_attempts > 0 else 0
                )
                print(f"Iteration {it + 1}/{n_iterations}, Swap rate: {swap_rate:.3f}")

        accept_rates = accept_counts / n_iterations
        swap_rate = len(swap_accepts) / swap_attempts if swap_attempts > 0 else 0

        return chains_history, swap_accepts, accept_rates, swap_rate


def plot_potential_and_target(gamma=4.0):
    """Plot the potential function and unnormalized target density"""
    x = np.linspace(-3, 3, 1000)
    U_x = gamma * (x**2 - 1) ** 2
    target = np.exp(-U_x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Potential function
    axes[0].plot(x, U_x, "b-", linewidth=2)
    axes[0].fill_between(x, 0, U_x, alpha=0.3)
    axes[0].set_xlabel("x", fontsize=12)
    axes[0].set_ylabel("U(x)", fontsize=12)
    axes[0].set_title(f"Potential: U(x) = {gamma}(x² - 1)²", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(-1, color="red", linestyle="--", alpha=0.5, label="Wells at x = ±1")
    axes[0].axvline(1, color="red", linestyle="--", alpha=0.5)
    axes[0].legend()

    # Target distribution
    axes[1].plot(x, target, "g-", linewidth=2)
    axes[1].fill_between(x, 0, target, alpha=0.3, color="green")
    axes[1].set_xlabel("x", fontsize=12)
    axes[1].set_ylabel("π(x)", fontsize=12)
    axes[1].set_title("Target: π(x) = exp(-U(x))", fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(-1, color="red", linestyle="--", alpha=0.5, label="Modes at x = ±1")
    axes[1].axvline(1, color="red", linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_chains_evolution(chains, temps, burn_in=1000, gamma=4.0):
    """Plot the evolution of all chains and final distributions"""
    n_chains = chains.shape[1]
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_chains))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Trace plots for all chains
    ax = axes[0, 0]
    for i in range(n_chains):
        ax.plot(
            chains[:, i],
            alpha=0.7,
            linewidth=0.5,
            color=colors[i],
            label=f"T={temps[i]:.2f}",
        )
    ax.axvline(burn_in, color="black", linestyle="--", label="Burn-in")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("x")
    ax.set_title("Chain Traces")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Histograms for each chain (after burn-in)
    ax = axes[0, 1]
    x_range = np.linspace(-3, 3, 100)
    for i in range(n_chains):
        counts, bins = np.histogram(chains[burn_in:, i], bins=50, density=True)
        centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(centers, counts, alpha=0.7, color=colors[i], label=f"T={temps[i]:.2f}")

    # Overlay true target
    true_target = np.exp(-gamma * (x_range**2 - 1) ** 2)
    true_target /= trapezoid(true_target, x_range)  # Normalize
    ax.plot(x_range, true_target, "k--", linewidth=2, label="True target")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Empirical Distributions")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Focus on cold chain
    ax = axes[1, 0]
    ax.plot(chains[:, 0], alpha=0.7, color="blue")
    ax.axvline(burn_in, color="red", linestyle="--", label="Burn-in")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("x")
    ax.set_title("Cold Chain (T=1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cold chain histogram with KDE
    ax = axes[1, 1]
    cold_samples = chains[burn_in:, 0]

    # Histogram
    counts, bins, _ = ax.hist(
        cold_samples, bins=50, density=True, alpha=0.5, color="blue", label="Samples"
    )

    # KDE
    kde = stats.gaussian_kde(cold_samples)
    x_kde = np.linspace(-3, 3, 200)
    ax.plot(x_kde, kde(x_kde), "b-", linewidth=2, label="KDE")

    # True target
    ax.plot(x_range, true_target, "r--", linewidth=2, label="True target")

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Cold Chain Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_swap_dynamics(chains, swap_accepts, temps):
    """Visualize chain swapping dynamics"""
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Swap history - Fixed to handle empty or mismatched data
    if swap_accepts and len(swap_accepts) > 0:
        swap_times = []
        swap_chains = []
        for swap in swap_accepts:
            if (
                len(swap) == 3
            ):  # Ensure swap has correct format (iteration, chain1, chain2)
                swap_times.append(swap[0])
                swap_chains.append((swap[1] + swap[2]) / 2.0)

        if swap_times and len(swap_times) == len(swap_chains):
            ax1.scatter(swap_times, swap_chains, alpha=0.5, s=10, c="red")

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Chain pair index")
    ax1.set_title("Successful Swaps")
    ax1.set_ylim(-0.5, len(temps) - 1.5)
    ax1.grid(True, alpha=0.3)

    # Temperature vs variance
    variances = [np.var(chains[:, i]) for i in range(len(temps))]
    ax2.plot(temps, variances, "o-", color="purple", linewidth=2, markersize=8)
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Chain Variance")
    ax2.set_title("Temperature Effect on Chain Variance")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_animation(chains, temps, interval=50, max_frames=500):
    """Create animation showing chain evolution"""
    n_chains = min(chains.shape[1], 5)  # Limit to 5 chains for clarity
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_chains))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Setup first subplot (chain positions)
    ax1.set_xlim(0, min(max_frames, len(chains)))
    ax1.set_ylim(-3, 3)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("x")
    ax1.set_title("Chain Evolution")
    ax1.grid(True, alpha=0.3)

    lines1 = []
    for i in range(n_chains):
        (line,) = ax1.plot(
            [], [], alpha=0.7, color=colors[i], label=f"T={temps[i]:.2f}"
        )
        lines1.append(line)
    ax1.legend(loc="upper right")

    # Setup second subplot (current histograms)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Density")
    ax2.set_title("Current Distributions")
    ax2.grid(True, alpha=0.3)

    def init():
        for line in lines1:
            line.set_data([], [])
        return lines1

    def update(frame):
        if frame < len(chains):
            # Update trace plots
            for i, line in enumerate(lines1):
                line.set_data(range(frame), chains[:frame, i])

            # Clear and redraw histograms
            ax2.clear()
            ax2.set_xlim(-3, 3)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel("x")
            ax2.set_ylabel("Density")
            ax2.set_title(f"Distributions at iteration {frame}")
            ax2.grid(True, alpha=0.3)

            if frame > 100:  # Start showing histograms after some burn-in
                for i in range(n_chains):
                    hist, bins = np.histogram(
                        chains[max(0, frame - 500) : frame, i], bins=30, density=True
                    )
                    centers = (bins[:-1] + bins[1:]) / 2
                    ax2.plot(centers, hist, alpha=0.7, color=colors[i])

        return lines1

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=min(max_frames, len(chains)),
        interval=interval,
        blit=False,
    )

    return anim


# Main execution and demonstration
if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)

    # Parameters
    gamma = 4.0  # Barrier height parameter
    n_chains = 5
    max_temp = 10.0
    n_iterations = 20000
    burn_in = 2000

    print(f"Double-well potential with γ = {gamma}")
    print(f"Running parallel tempering with {n_chains} chains")
    print(f"Temperature range: 1.0 to {max_temp}")
    print("-" * 50)

    # Initialize and run parallel tempering
    sampler = ParallelTemperingDoubleWell(
        gamma=gamma, n_chains=n_chains, max_temp=max_temp
    )

    print(f"Temperature ladder: {sampler.temps}")
    print("-" * 50)

    # Run simulation
    chains, swap_accepts, accept_rates, swap_rate = sampler.run(
        n_iterations=n_iterations,
        step_size=0.5,
        swap_interval=10,
        initial_x=0.0,
        verbose=True,
    )

    print("-" * 50)
    print(f"Acceptance rates by chain: {accept_rates}")
    print(f"Overall swap acceptance rate: {swap_rate:.3f}")
    print(f"Total successful swaps: {len(swap_accepts)}")

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. Potential and target distribution
    fig1 = plot_potential_and_target(gamma)
    plt.show()

    # 2. Chain evolution and distributions
    fig2 = plot_chains_evolution(chains, sampler.temps, burn_in=burn_in, gamma=gamma)
    plt.show()

    # 3. Swap dynamics
    fig3 = plot_swap_dynamics(chains, swap_accepts, sampler.temps)
    plt.show()

    # 4. Convergence diagnostics
    fig4, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Autocorrelation for cold chain
    ax = axes[0]
    cold_chain = chains[burn_in:, 0]
    lags = range(0, min(100, len(cold_chain) // 4))
    autocorr = [1.0]  # lag 0 is always 1
    for lag in range(1, len(lags)):
        if lag < len(cold_chain):
            autocorr.append(np.corrcoef(cold_chain[:-lag], cold_chain[lag:])[0, 1])

    ax.plot(lags[: len(autocorr)], autocorr, "b-", linewidth=2)
    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Cold Chain Autocorrelation")
    ax.grid(True, alpha=0.3)

    # Effective sample size estimate
    ax = axes[1]
    ess_window = 1000
    ess_estimates = []
    positions = []

    for i in range(burn_in + ess_window, len(chains), 100):
        subset = chains[i - ess_window : i, 0]
        # Simple ESS estimate based on autocorrelation
        autocorr_subset = []
        for j in range(1, min(50, len(subset) // 4)):
            if j < len(subset):
                autocorr_subset.append(np.corrcoef(subset[:-j], subset[j:])[0, 1])

        if autocorr_subset:
            first_negative = next(
                (j for j, ac in enumerate(autocorr_subset) if ac < 0.05),
                len(autocorr_subset),
            )
            ess = ess_window / (1 + 2 * sum(autocorr_subset[:first_negative]))
            ess_estimates.append(ess)
            positions.append(i)

    if positions:
        ax.plot(positions, ess_estimates, "g-", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Effective Sample Size")
    ax.set_title("ESS Evolution (Cold Chain)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("\nSummary Statistics for Cold Chain (after burn-in):")
    cold_samples = chains[burn_in:, 0]
    print(f"Mean: {np.mean(cold_samples):.4f}")
    print(f"Std: {np.std(cold_samples):.4f}")
    print(f"Min: {np.min(cold_samples):.4f}")
    print(f"Max: {np.max(cold_samples):.4f}")

    # Check for mode switching
    left_well = np.sum(cold_samples < 0) / len(cold_samples)
    right_well = np.sum(cold_samples > 0) / len(cold_samples)
    print("\nMode occupation:")
    print(f"Left well (x < 0): {left_well:.2%}")
    print(f"Right well (x > 0): {right_well:.2%}")

    # Optional: Create and save animation
    # print("\nCreating animation...")
    # anim = create_animation(chains[:5000], sampler.temps, interval=50, max_frames=500)
    # anim.save('parallel_tempering.mp4', writer='ffmpeg', fps=30)
    # # For Jupyter: HTML(anim.to_jshtml())
