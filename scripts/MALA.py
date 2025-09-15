import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


class MALA:
    """
    Metropolis-adjusted Langevin Algorithm (MALA) sampler.

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
            Step size parameter (also called epsilon or tau)
        """
        self.log_target = log_target
        self.grad_log_target = grad_log_target
        self.step_size = step_size

    def proposal_mean(self, x):
        """
        Compute the mean of the Langevin proposal distribution.

        The proposal follows: x_new ~ N(x + (step_size^2/2) * grad_log_p(x), step_size^2 * I)
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
            return x_proposed, True
        else:
            return x_current, False

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
            x_current, accepted = self.accept_reject(x_current, x_proposed)
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


# ============================================================================
# Example 1: Sampling from a 2D Gaussian distribution
# ============================================================================


def example_gaussian():
    """Sample from a 2D Gaussian distribution."""
    print("Example 1: Sampling from 2D Gaussian")
    print("-" * 40)

    # Define target distribution: N(mean, cov)
    mean = np.array([2.0, -1.0])
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    cov_inv = np.linalg.inv(cov)

    def log_target(x):
        """Log probability of multivariate Gaussian."""
        diff = x - mean
        return -0.5 * diff @ cov_inv @ diff

    def grad_log_target(x):
        """Gradient of log probability."""
        diff = x - mean
        return -cov_inv @ diff

    # Initialize sampler
    sampler = MALA(log_target, grad_log_target, step_size=0.5)

    # Generate samples
    x0 = np.array([0.0, 0.0])
    samples, acc_rate = sampler.sample(x0, n_samples=5000, burn_in=500, verbose=True)

    print(f"\nAcceptance rate: {acc_rate:.3f}")
    print(f"Sample mean: {samples.mean(axis=0)}")
    print(f"True mean: {mean}")
    print(f"Sample covariance:\n{np.cov(samples.T)}")
    print(f"True covariance:\n{cov}")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot of samples
    axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")
    axes[0].set_title("MALA Samples from 2D Gaussian")
    axes[0].set_aspect("equal")

    # Trace plots
    axes[1].plot(samples[:1000, 0], alpha=0.7, label="X1")
    axes[1].plot(samples[:1000, 1], alpha=0.7, label="X2")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Trace Plot (first 1000 samples)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ============================================================================
# Example 2: Sampling from a banana-shaped distribution
# ============================================================================


def example_banana():
    """Sample from Rosenbrock (banana-shaped) distribution."""
    print("\nExample 2: Sampling from Banana-shaped Distribution")
    print("-" * 50)

    # Parameters for the banana distribution
    a = 1.0
    b = 100.0

    def log_target(x):
        """Log probability of banana distribution."""
        x1, x2 = x[0], x[1]
        return -b * (x2 - x1**2) ** 2 - (a - x1) ** 2

    def grad_log_target(x):
        """Gradient of log probability."""
        x1, x2 = x[0], x[1]
        grad = np.zeros(2)
        grad[0] = 4 * b * x1 * (x2 - x1**2) + 2 * (a - x1)
        grad[1] = -2 * b * (x2 - x1**2)
        return grad

    # Initialize sampler with smaller step size for this challenging distribution
    sampler = MALA(log_target, grad_log_target, step_size=0.1)

    # Generate samples
    x0 = np.array([0.0, 0.0])
    samples, acc_rate = sampler.sample(
        x0, n_samples=10000, burn_in=1000, thin=2, verbose=True
    )

    print(f"\nAcceptance rate: {acc_rate:.3f}")
    print(f"Sample mean: {samples.mean(axis=0)}")
    print(f"Sample std: {samples.std(axis=0)}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Scatter plot of samples
    axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")
    axes[0].set_title("MALA Samples from Banana Distribution")
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-1, 4)

    # Contour plot of true distribution
    x1_grid = np.linspace(-2, 2, 100)
    x2_grid = np.linspace(-1, 4, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = np.exp(log_target(np.array([X1[i, j], X2[i, j]])))

    axes[1].contour(X1, X2, Z, levels=10)
    axes[1].set_xlabel("X1")
    axes[1].set_ylabel("X2")
    axes[1].set_title("True Distribution Contours")

    # Histogram of X1 marginal
    axes[2].hist(samples[:, 0], bins=50, density=True, alpha=0.7)
    axes[2].set_xlabel("X1")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Marginal Distribution of X1")

    plt.tight_layout()
    plt.show()


# ============================================================================
# Example 3: Bayesian logistic regression
# ============================================================================


def example_bayesian_logistic():
    """Sample from posterior of Bayesian logistic regression."""
    print("\nExample 3: Bayesian Logistic Regression")
    print("-" * 40)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 2

    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([1.5, -2.0])
    logits = X @ true_beta
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)

    print(f"Generated {n_samples} samples with {n_features} features")
    print(f"True coefficients: {true_beta}")

    # Prior parameters (Gaussian prior)
    prior_mean = np.zeros(n_features)
    prior_precision = 0.1 * np.eye(n_features)  # Weak prior

    def sigmoid(z):
        """Stable sigmoid function."""
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def log_target(beta):
        """Log posterior = log likelihood + log prior."""
        # Log likelihood
        logits = X @ beta
        log_lik = np.sum(y * logits - np.log(1 + np.exp(logits)))

        # Log prior
        diff = beta - prior_mean
        log_prior = -0.5 * diff @ prior_precision @ diff

        return log_lik + log_prior

    def grad_log_target(beta):
        """Gradient of log posterior."""
        # Gradient of log likelihood
        probs = sigmoid(X @ beta)
        grad_log_lik = X.T @ (y - probs)

        # Gradient of log prior
        grad_log_prior = -prior_precision @ (beta - prior_mean)

        return grad_log_lik + grad_log_prior

    # Initialize sampler
    sampler = MALA(log_target, grad_log_target, step_size=0.05)

    # Generate samples
    beta0 = np.zeros(n_features)
    samples, acc_rate = sampler.sample(
        beta0, n_samples=5000, burn_in=1000, verbose=True
    )

    print(f"\nAcceptance rate: {acc_rate:.3f}")
    print(f"Posterior mean: {samples.mean(axis=0)}")
    print(f"Posterior std: {samples.std(axis=0)}")
    print(f"True coefficients: {true_beta}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Scatter plot of samples
    axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
    axes[0].axhline(y=true_beta[1], color="r", linestyle="--", label="True β₂")
    axes[0].axvline(x=true_beta[0], color="r", linestyle="--", label="True β₁")
    axes[0].set_xlabel("β₁")
    axes[0].set_ylabel("β₂")
    axes[0].set_title("Posterior Samples")
    axes[0].legend()

    # Marginal distributions
    for i in range(n_features):
        axes[i + 1].hist(samples[:, i], bins=50, density=True, alpha=0.7)
        axes[i + 1].axvline(
            x=true_beta[i], color="r", linestyle="--", label=f"True β_{i + 1}"
        )
        axes[i + 1].axvline(
            x=samples[:, i].mean(), color="g", linestyle="-", label="Posterior mean"
        )
        axes[i + 1].set_xlabel(f"β_{i + 1}")
        axes[i + 1].set_ylabel("Density")
        axes[i + 1].set_title(f"Marginal Posterior of β_{i + 1}")
        axes[i + 1].legend()

    plt.tight_layout()
    plt.show()


# ============================================================================
# Adaptive MALA with automatic step size tuning
# ============================================================================


class AdaptiveMALA(MALA):
    """
    MALA with adaptive step size tuning during burn-in.
    """

    def sample_adaptive(
        self,
        x0,
        n_samples,
        burn_in=1000,
        target_acc_rate=0.574,
        adaptation_interval=100,
        thin=1,
        verbose=False,
    ):
        """
        Sample with adaptive step size tuning.

        The optimal acceptance rate for MALA is approximately 0.574
        for Gaussian targets (Roberts & Rosenthal, 1998).
        """
        x0 = np.asarray(x0)
        total_iterations = burn_in + n_samples * thin

        samples = []
        x_current = x0.copy()
        n_accepted = 0
        n_accepted_recent = 0

        for i in range(total_iterations):
            # Propose and accept/reject
            x_proposed = self.propose(x_current)
            x_current, accepted = self.accept_reject(x_current, x_proposed)
            n_accepted += accepted
            n_accepted_recent += accepted

            # Adapt step size during burn-in
            if i < burn_in and (i + 1) % adaptation_interval == 0:
                recent_acc_rate = n_accepted_recent / adaptation_interval

                # Adjust step size based on acceptance rate
                if recent_acc_rate < target_acc_rate - 0.05:
                    self.step_size *= 0.9  # Decrease step size
                elif recent_acc_rate > target_acc_rate + 0.05:
                    self.step_size *= 1.1  # Increase step size

                if verbose:
                    print(
                        f"Adaptation at iter {i + 1}: acc_rate={recent_acc_rate:.3f}, "
                        f"step_size={self.step_size:.4f}"
                    )

                n_accepted_recent = 0

            # Store sample if past burn-in
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(x_current.copy())

            if verbose and (i + 1) % 1000 == 0 and i >= burn_in:
                print(
                    f"Iteration {i + 1}/{total_iterations}, "
                    f"Acceptance rate: {n_accepted / (i + 1):.3f}"
                )

        samples = np.array(samples)
        acceptance_rate = n_accepted / total_iterations

        return samples, acceptance_rate, self.step_size


def example_adaptive():
    """Example using adaptive MALA."""
    print("\nExample 4: Adaptive MALA")
    print("-" * 40)

    # Use a challenging target: mixture of Gaussians
    def log_target(x):
        """Log probability of mixture of two Gaussians."""
        # Component 1: N([-2, 0], I)
        log_p1 = -0.5 * np.sum((x - np.array([-2, 0])) ** 2)
        # Component 2: N([2, 0], I)
        log_p2 = -0.5 * np.sum((x - np.array([2, 0])) ** 2)
        # Mixture with equal weights
        return np.log(0.5 * np.exp(log_p1) + 0.5 * np.exp(log_p2))

    def grad_log_target(x):
        """Gradient of log probability."""
        # Component probabilities
        diff1 = x - np.array([-2, 0])
        diff2 = x - np.array([2, 0])
        p1 = np.exp(-0.5 * np.sum(diff1**2))
        p2 = np.exp(-0.5 * np.sum(diff2**2))

        # Weighted gradients
        w1 = p1 / (p1 + p2)
        w2 = p2 / (p1 + p2)

        return -w1 * diff1 - w2 * diff2

    # Initialize adaptive sampler with initial step size
    sampler = AdaptiveMALA(log_target, grad_log_target, step_size=1.0)

    # Generate samples with adaptation
    x0 = np.array([0.0, 0.0])
    samples, acc_rate, final_step = sampler.sample_adaptive(
        x0, n_samples=5000, burn_in=2000, verbose=True
    )

    print(f"\nFinal acceptance rate: {acc_rate:.3f}")
    print(f"Final step size: {final_step:.4f}")
    print(f"Sample mean: {samples.mean(axis=0)}")

    # Plot results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Adaptive MALA: Mixture of Gaussians")

    plt.subplot(1, 2, 2)
    plt.hist(samples[:, 0], bins=50, density=True, alpha=0.7)
    plt.xlabel("X1")
    plt.ylabel("Density")
    plt.title("Marginal Distribution of X1")

    plt.tight_layout()
    plt.show()


# ============================================================================
# Run all examples
# ============================================================================

if __name__ == "__main__":
    # Run examples
    example_gaussian()
    example_banana()
    example_bayesian_logistic()
    example_adaptive()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
