import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


class MALA:
    """
    Basic Metropolis-adjusted Langevin Algorithm (MALA) sampler.

    MALA uses gradient information to propose new states, making it more efficient
    than random walk Metropolis-Hastings for smooth target distributions.
    """

    def __init__(self, log_target, grad_log_target, step_size=0.1):
        """
        Initialize MALA sampler.

        Parameters:
        -----------
        log_target : callable
            Function that computes log probability of target distribution
        grad_log_target : callable
            Function that computes gradient of log probability
        step_size : float
            Step size parameter (epsilon)
        """
        self.log_target = log_target
        self.grad_log_target = grad_log_target
        self.step_size = step_size

    def proposal_mean(self, x):
        """
        Compute the mean of the Langevin proposal distribution.

        The proposal follows: x_new ~ N(x + (ε²/2) * ∇log p(x), ε² * I)
        """
        return x + 0.5 * self.step_size**2 * self.grad_log_target(x)

    def log_proposal_density(self, x_new, x_old):
        """
        Compute log density of the proposal distribution q(x_new | x_old).
        """
        mean = self.proposal_mean(x_old)
        cov = self.step_size**2 * np.eye(len(x_old))
        return multivariate_normal.logpdf(x_new, mean, cov)

    def propose(self, x):
        """
        Generate a proposal using Langevin dynamics.
        """
        mean = self.proposal_mean(x)
        noise = np.random.normal(0, self.step_size, size=x.shape)
        return mean + noise

    def accept_reject(self, x_current, x_proposed):
        """
        Compute acceptance probability and decide whether to accept proposal.
        """
        # Compute log acceptance ratio
        log_alpha = (
            self.log_target(x_proposed)
            - self.log_target(x_current)
            + self.log_proposal_density(x_current, x_proposed)
            - self.log_proposal_density(x_proposed, x_current)
        )

        # Accept with probability min(1, exp(log_alpha))
        alpha = min(1.0, np.exp(log_alpha))

        if np.random.uniform() < alpha:
            return x_proposed, True, alpha
        else:
            return x_current, False, alpha

    def sample(self, x0, n_samples, burn_in=0, thin=1, verbose=False):
        """
        Generate samples using MALA.

        Parameters:
        -----------
        x0 : array-like
            Initial state
        n_samples : int
            Number of samples to generate
        burn_in : int
            Number of burn-in samples to discard
        thin : int
            Thinning parameter (keep every thin-th sample)
        verbose : bool
            Print progress information

        Returns:
        --------
        samples : array
            Generated samples
        acceptance_rate : float
            Proportion of accepted proposals
        """
        x0 = np.asarray(x0)
        total_iterations = burn_in + n_samples * thin

        samples = []
        x_current = x0.copy()
        n_accepted = 0

        for i in range(total_iterations):
            # Propose new state
            x_proposed = self.propose(x_current)

            # Accept or reject
            x_current, accepted, _ = self.accept_reject(x_current, x_proposed)
            n_accepted += accepted

            # Store sample if past burn-in and at thinning interval
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(x_current.copy())

            if verbose and (i + 1) % 1000 == 0:
                print(
                    f"Iteration {i + 1}/{total_iterations}, "
                    f"Acceptance rate: {n_accepted / (i + 1):.3f}"
                )

        samples = np.array(samples)
        acceptance_rate = n_accepted / total_iterations

        return samples, acceptance_rate


class RobbinsMonroMALA(MALA):
    """
    MALA with Robbins-Monro adaptive step size tuning.

    Based on:
    Andrieu, C., & Thoms, J. (2008). A tutorial on adaptive MCMC.
    Statistics and Computing, 18(4), 343-373.

    The Robbins-Monro stochastic approximation adapts the log step size using:
    log(ε_{n+1}) = log(ε_n) + γ_n * (α_n - α_target)

    where:
    - ε_n is the step size at iteration n
    - γ_n is the adaptation rate (decreasing sequence)
    - α_n is the acceptance indicator (0 or 1)
    - α_target is the target acceptance rate (0.574 for MALA)
    """

    def __init__(
        self,
        log_target,
        grad_log_target,
        initial_step_size=1.0,
        target_accept_rate=0.574,
        gamma_decay=0.66,
        gamma_scale=1.0,
        adaptation_offset=10,
    ):
        """
        Initialize Robbins-Monro adaptive MALA.

        Parameters:
        -----------
        log_target : callable
            Function that computes log probability
        grad_log_target : callable
            Function that computes gradient of log probability
        initial_step_size : float
            Initial step size (ε_0)
        target_accept_rate : float
            Target acceptance rate (default 0.574 from Roberts & Rosenthal 1998)
        gamma_decay : float
            Decay rate κ for adaptation: γ_n = γ_scale / (n + offset)^κ
            Should be in (0.5, 1] for theoretical convergence
        gamma_scale : float
            Scale factor for adaptation rate
        adaptation_offset : int
            Offset for adaptation schedule to avoid large initial updates
        """
        super().__init__(log_target, grad_log_target, initial_step_size)

        self.target_accept_rate = target_accept_rate
        self.gamma_decay = gamma_decay
        self.gamma_scale = gamma_scale
        self.adaptation_offset = adaptation_offset

        # Track log step size for numerical stability
        self.log_step_size = np.log(initial_step_size)

        # Bounds for step size to maintain numerical stability
        self.log_step_min = np.log(1e-5)
        self.log_step_max = np.log(100)

    def get_adaptation_rate(self, iteration):
        """
        Compute the adaptation rate γ_n at iteration n.

        Uses γ_n = γ_scale / (n + offset)^κ
        """
        return (
            self.gamma_scale / (iteration + self.adaptation_offset) ** self.gamma_decay
        )

    def update_step_size(self, iteration, accepted):
        """
        Update step size using Robbins-Monro stochastic approximation.

        Parameters:
        -----------
        iteration : int
            Current iteration number (starts from 0)
        accepted : bool
            Whether the last proposal was accepted
        """
        # Get adaptation rate for this iteration
        gamma_n = self.get_adaptation_rate(iteration)

        # Update log step size
        # α_n is 1 if accepted, 0 if rejected
        alpha_n = float(accepted)
        self.log_step_size += gamma_n * (alpha_n - self.target_accept_rate)

        # Enforce bounds for numerical stability
        self.log_step_size = np.clip(
            self.log_step_size, self.log_step_min, self.log_step_max
        )

        # Update actual step size
        self.step_size = np.exp(self.log_step_size)

    def sample_adaptive(
        self, x0, n_samples, burn_in=1000, thin=1, verbose=False, track_step_size=False
    ):
        """
        Sample with Robbins-Monro adaptive step size tuning.

        Parameters:
        -----------
        x0 : array-like
            Initial state
        n_samples : int
            Number of samples to generate (after burn-in)
        burn_in : int
            Number of burn-in iterations with adaptation
        thin : int
            Thinning parameter
        verbose : bool
            Print progress information
        track_step_size : bool
            Whether to track step size history

        Returns:
        --------
        samples : array
            Generated samples
        info : dict
            Dictionary containing:
            - 'acceptance_rate': overall acceptance rate
            - 'final_step_size': final adapted step size
            - 'step_size_history': history of step sizes (if track_step_size=True)
            - 'acceptance_probs': history of acceptance probabilities
        """
        x0 = np.asarray(x0)
        total_iterations = burn_in + n_samples * thin

        samples = []
        x_current = x0.copy()
        n_accepted = 0

        step_size_history = [] if track_step_size else None
        acceptance_probs = []

        for i in range(total_iterations):
            # Propose new state
            x_proposed = self.propose(x_current)

            # Accept or reject
            x_current, accepted, alpha = self.accept_reject(x_current, x_proposed)
            n_accepted += accepted
            acceptance_probs.append(alpha)

            # Adapt step size during burn-in using Robbins-Monro
            if i < burn_in:
                self.update_step_size(i, accepted)

            # Track step size if requested
            if track_step_size:
                step_size_history.append(self.step_size)

            # Store sample if past burn-in and at thinning interval
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(x_current.copy())

            # Progress reporting
            if verbose:
                if i < burn_in and (i + 1) % 100 == 0:
                    recent_acc_rate = np.mean([a for a in acceptance_probs[-100:]])
                    print(
                        f"Burn-in {i + 1}/{burn_in}: "
                        f"step_size={self.step_size:.4f}, "
                        f"recent_acc_rate={recent_acc_rate:.3f}"
                    )
                elif (i + 1) % 1000 == 0:
                    print(
                        f"Iteration {i + 1}/{total_iterations}, "
                        f"Acceptance rate: {n_accepted / (i + 1):.3f}"
                    )

        samples = np.array(samples)
        acceptance_rate = n_accepted / total_iterations

        info = {
            "acceptance_rate": acceptance_rate,
            "final_step_size": self.step_size,
            "step_size_history": step_size_history,
            "acceptance_probs": acceptance_probs,
        }

        return samples, info


# ============================================================================
# Example 1: Gaussian target with adaptation diagnostics
# ============================================================================


def example_gaussian_adaptation():
    """
    Demonstrate Robbins-Monro adaptation on a 2D Gaussian.
    Shows convergence of step size and acceptance rate.
    """
    print("=" * 60)
    print("Example 1: Robbins-Monro Adaptation on 2D Gaussian")
    print("=" * 60)

    # Define target: N(mean, cov)
    mean = np.array([2.0, -1.0])
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    cov_inv = np.linalg.inv(cov)

    def log_target(x):
        diff = x - mean
        return -0.5 * diff @ cov_inv @ diff

    def grad_log_target(x):
        diff = x - mean
        return -cov_inv @ diff

    # Initialize adaptive sampler with poor initial step size
    sampler = RobbinsMonroMALA(
        log_target,
        grad_log_target,
        initial_step_size=5.0,  # Intentionally too large
        target_accept_rate=0.574,
        gamma_decay=0.66,  # As recommended by Andrieu & Thoms
        gamma_scale=1.0,
    )

    # Run adaptive sampling
    x0 = np.array([0.0, 0.0])
    samples, info = sampler.sample_adaptive(
        x0, n_samples=5000, burn_in=2000, verbose=True, track_step_size=True
    )

    print(f"\nFinal acceptance rate: {info['acceptance_rate']:.3f}")
    print(f"Target acceptance rate: {sampler.target_accept_rate:.3f}")
    print(f"Final step size: {info['final_step_size']:.4f}")
    print(f"Sample mean: {samples.mean(axis=0)}")
    print(f"True mean: {mean}")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Step size evolution
    axes[0, 0].plot(info["step_size_history"][:2000])
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Step Size")
    axes[0, 0].set_title("Step Size Adaptation (Burn-in)")
    axes[0, 0].grid(True, alpha=0.3)

    # Running acceptance rate
    window = 100
    running_acc = np.convolve(
        info["acceptance_probs"][:2000], np.ones(window) / window, mode="valid"
    )
    axes[0, 1].plot(running_acc)
    axes[0, 1].axhline(y=0.574, color="r", linestyle="--", label="Target (0.574)")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Running Acceptance Rate")
    axes[0, 1].set_title(f"Acceptance Rate (window={window})")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Adaptation rate decay
    iterations = np.arange(1, 2001)
    gamma_values = [sampler.get_adaptation_rate(i) for i in iterations]
    axes[0, 2].loglog(iterations, gamma_values)
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("Adaptation Rate γ_n")
    axes[0, 2].set_title(f"Robbins-Monro Rate Decay (κ={sampler.gamma_decay})")
    axes[0, 2].grid(True, alpha=0.3)

    # Samples scatter plot
    axes[1, 0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)
    axes[1, 0].set_xlabel("X1")
    axes[1, 0].set_ylabel("X2")
    axes[1, 0].set_title("Samples from 2D Gaussian")
    axes[1, 0].set_aspect("equal")

    # Trace plots
    axes[1, 1].plot(samples[:1000, 0], alpha=0.7, label="X1")
    axes[1, 1].plot(samples[:1000, 1], alpha=0.7, label="X2")
    axes[1, 1].set_xlabel("Sample")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_title("Trace Plot")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # ACF plot
    from statsmodels.tsa.stattools import acf

    lags = 50
    acf_x1 = acf(samples[:, 0], nlags=lags)
    acf_x2 = acf(samples[:, 1], nlags=lags)
    axes[1, 2].plot(acf_x1, label="X1")
    axes[1, 2].plot(acf_x2, label="X2")
    axes[1, 2].set_xlabel("Lag")
    axes[1, 2].set_ylabel("Autocorrelation")
    axes[1, 2].set_title("Autocorrelation Function")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Example 2: Comparison of different gamma_decay values
# ============================================================================


def example_gamma_comparison():
    """
    Compare different values of gamma_decay parameter.
    Theory suggests κ ∈ (0.5, 1] with trade-offs between speed and stability.
    """
    print("\n" + "=" * 60)
    print("Example 2: Effect of γ Decay Rate")
    print("=" * 60)

    # Banana-shaped target for challenging adaptation
    a, b = 1.0, 100.0

    def log_target(x):
        x1, x2 = x[0], x[1]
        return -b * (x2 - x1**2) ** 2 - (a - x1) ** 2

    def grad_log_target(x):
        x1, x2 = x[0], x[1]
        grad = np.zeros(2)
        grad[0] = 4 * b * x1 * (x2 - x1**2) + 2 * (a - x1)
        grad[1] = -2 * b * (x2 - x1**2)
        return grad

    # Test different decay rates
    decay_rates = [0.51, 0.66, 0.8, 0.99]
    colors = ["blue", "green", "orange", "red"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for decay, color in zip(decay_rates, colors):
        print(f"\nTesting γ decay = {decay}")

        sampler = RobbinsMonroMALA(
            log_target,
            grad_log_target,
            initial_step_size=1.0,
            gamma_decay=decay,
            gamma_scale=1.0,
        )

        x0 = np.array([0.0, 0.0])
        samples, info = sampler.sample_adaptive(
            x0, n_samples=3000, burn_in=3000, track_step_size=True
        )

        print(f"  Final step size: {info['final_step_size']:.4f}")
        print(f"  Acceptance rate: {info['acceptance_rate']:.3f}")

        # Plot step size evolution
        axes[0, 0].plot(
            info["step_size_history"][:3000], label=f"κ={decay}", color=color, alpha=0.7
        )

        # Plot running acceptance rate
        window = 100
        running_acc = np.convolve(
            info["acceptance_probs"][:3000], np.ones(window) / window, mode="valid"
        )
        axes[0, 1].plot(running_acc, color=color, alpha=0.7)

        # Plot samples
        axes[1, 0].scatter(
            samples[::10, 0], samples[::10, 1], alpha=0.3, s=1, color=color
        )

        # Plot adaptation rates
        iterations = np.logspace(0, 3.5, 100)
        gamma_values = [
            sampler.gamma_scale / (i + sampler.adaptation_offset) ** decay
            for i in iterations
        ]
        axes[1, 1].loglog(iterations, gamma_values, label=f"κ={decay}", color=color)

    # Formatting
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Step Size")
    axes[0, 0].set_title("Step Size Evolution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].axhline(y=0.574, color="black", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Running Accept Rate")
    axes[0, 1].set_title("Acceptance Rate Convergence")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("X1")
    axes[1, 0].set_ylabel("X2")
    axes[1, 0].set_title("Samples from Banana Distribution")
    axes[1, 0].set_xlim(-2, 2)
    axes[1, 0].set_ylim(-1, 4)

    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("γ_n")
    axes[1, 1].set_title("Adaptation Rate Decay")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Example 3: Multimodal target with adaptation
# ============================================================================


def example_multimodal():
    """
    Test Robbins-Monro adaptation on a challenging multimodal target.
    """
    print("\n" + "=" * 60)
    print("Example 3: Multimodal Target")
    print("=" * 60)

    # Mixture of three Gaussians
    def log_target(x):
        # Component 1: N([-3, 0], 0.5*I)
        log_p1 = -np.sum((x - np.array([-3, 0])) ** 2) / (2 * 0.5)
        # Component 2: N([0, 3], 0.5*I)
        log_p2 = -np.sum((x - np.array([0, 3])) ** 2) / (2 * 0.5)
        # Component 3: N([3, -1], 0.5*I)
        log_p3 = -np.sum((x - np.array([3, -1])) ** 2) / (2 * 0.5)

        # Log-sum-exp for numerical stability
        log_probs = np.array([log_p1, log_p2, log_p3])
        max_log = np.max(log_probs)
        return max_log + np.log(np.sum(np.exp(log_probs - max_log)) / 3)

    def grad_log_target(x):
        # Compute weights for each component
        var = 0.5
        means = [np.array([-3, 0]), np.array([0, 3]), np.array([3, -1])]

        diffs = [x - m for m in means]
        log_probs = [-np.sum(d**2) / (2 * var) for d in diffs]

        # Softmax weights
        max_log = np.max(log_probs)
        exp_probs = np.exp(np.array(log_probs) - max_log)
        weights = exp_probs / np.sum(exp_probs)

        # Weighted gradient
        grad = np.zeros(2)
        for w, d in zip(weights, diffs):
            grad -= w * d / var
        return grad

    # Run adaptive sampling with different initial conditions
    initial_points = [
        np.array([0.0, 0.0]),
        np.array([-3.0, 0.0]),
        np.array([3.0, -1.0]),
    ]
    all_samples = []

    for i, x0 in enumerate(initial_points):
        print(f"\nStarting from {x0}")

        sampler = RobbinsMonroMALA(
            log_target,
            grad_log_target,
            initial_step_size=2.0,
            target_accept_rate=0.574,
            gamma_decay=0.66,
        )

        samples, info = sampler.sample_adaptive(
            x0, n_samples=5000, burn_in=2000, verbose=False, track_step_size=True
        )

        all_samples.append(samples)
        print(f"  Final step size: {info['final_step_size']:.4f}")
        print(f"  Acceptance rate: {info['acceptance_rate']:.3f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["blue", "green", "red"]
    for samples, color, x0 in zip(all_samples, colors, initial_points):
        for ax in axes:
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                alpha=0.3,
                s=1,
                color=color,
                label=f"Start: [{x0[0]:.0f}, {x0[1]:.0f}]",
            )

    # Add contours of true distribution
    x_grid = np.linspace(-6, 6, 100)
    y_grid = np.linspace(-4, 6, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = np.exp(log_target(np.array([X[i, j], Y[i, j]])))

    for ax in axes:
        ax.contour(X, Y, Z, levels=10, colors="gray", alpha=0.3)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_aspect("equal")

    axes[0].set_title("All Chains Combined")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_title("True Distribution Contours")
    axes[1].set_xlim(-6, 6)
    axes[1].set_ylim(-4, 6)

    # Marginal distributions
    all_combined = np.vstack(all_samples)
    axes[2].hist(
        all_combined[:, 0], bins=50, density=True, alpha=0.7, orientation="horizontal"
    )
    axes[2].set_xlabel("Density")
    axes[2].set_title("Marginal of X1")
    axes[2].set_ylim(-6, 6)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Run examples
# ============================================================================

if __name__ == "__main__":
    # Run all examples
    example_gaussian_adaptation()
    example_gamma_comparison()
    example_multimodal()

    print("\n" + "=" * 60)
    print("Robbins-Monro Adaptive MALA Examples Complete!")
    print("=" * 60)
    print("\nReference:")
    print("Andrieu, C., & Thoms, J. (2008). A tutorial on adaptive MCMC.")
    print("Statistics and Computing, 18(4), 343-373.")
    print("\nKey features of this implementation:")
    print("- Adapts log(step_size) for numerical stability")
    print("- Uses γ_n = γ_scale / (n + offset)^κ adaptation schedule")
    print("- κ ∈ (0.5, 1] ensures theoretical convergence")
    print("- Targets optimal 0.574 acceptance rate from Roberts & Rosenthal (1998)")
    print("=" * 60)
