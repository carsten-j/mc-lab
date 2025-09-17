import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import laplace


class SqueezeRejectionSampler:
    """
    Implements squeeze rejection sampling for sampling from standard normal
    using Laplace distribution as proposal.
    """

    def __init__(self):
        # M = sqrt(e) as calculated in the solution
        self.M = np.sqrt(np.e)
        self.pi_tilde_evals = 0  # Counter for expensive evaluations
        self.total_proposals = 0  # Total number of proposals
        self.step_b_accepts = 0  # Accepts in step (b)
        self.step_c_accepts = 0  # Accepts in step (c)

    def h(self, x):
        """Lower bound function: h(x) = max(0, 1 - x^2/2)"""
        result = 1 - x**2 / 2
        return np.maximum(0, result)

    def pi_tilde(self, x):
        """Target unnormalized density: exp(-x^2/2)"""
        self.pi_tilde_evals += 1
        return np.exp(-(x**2) / 2)

    def q_tilde(self, x):
        """Proposal unnormalized density: exp(-|x|)"""
        return np.exp(-np.abs(x))

    def sample_laplace(self):
        """Sample from Laplace(0, 1) distribution"""
        return laplace.rvs(size=1)

    def sample_one(self):
        """Generate one sample using squeeze rejection algorithm"""
        while True:
            self.total_proposals += 1

            # Step (a): Draw X ~ q, U ~ U[0,1]
            x = self.sample_laplace()
            u = np.random.uniform()

            # Step (b): Check if U <= h(X)/(M*q_tilde(X))
            h_x = self.h(x)
            q_tilde_x = self.q_tilde(x)

            if u <= h_x / (self.M * q_tilde_x):
                # Accept without evaluating pi_tilde
                self.step_b_accepts += 1
                return x

            # Step (c): Need to evaluate pi_tilde
            v = np.random.uniform()
            pi_tilde_x = self.pi_tilde(x)

            if v <= (pi_tilde_x - h_x) / (self.M * q_tilde_x - h_x):
                # Accept after evaluating pi_tilde
                self.step_c_accepts += 1
                return x
            # Otherwise reject and continue

    def sample(self, n):
        """Generate n samples"""
        samples = np.array([self.sample_one() for _ in range(n)])
        return samples

    def get_statistics(self):
        """Return sampling statistics"""
        if self.total_proposals == 0:
            return {}

        prob_not_eval = 1 - (self.pi_tilde_evals / self.total_proposals)
        prob_accept_b = self.step_b_accepts / self.total_proposals
        prob_accept_c = self.step_c_accepts / self.total_proposals
        acceptance_rate = (
            self.step_b_accepts + self.step_c_accepts
        ) / self.total_proposals

        return {
            "total_proposals": self.total_proposals,
            "pi_tilde_evaluations": self.pi_tilde_evals,
            "step_b_accepts": self.step_b_accepts,
            "step_c_accepts": self.step_c_accepts,
            "prob_not_evaluating_pi": prob_not_eval,
            "prob_accept_step_b": prob_accept_b,
            "prob_accept_step_c": prob_accept_c,
            "overall_acceptance_rate": acceptance_rate,
            "theoretical_prob_not_eval": 2 * np.sqrt(2) / (3 * np.sqrt(np.e)),
        }


def verify_implementation(n_samples=50000, show_plots=True):
    """Verify the implementation with empirical results"""

    # Initialize sampler
    sampler = SqueezeRejectionSampler()

    # Generate samples
    print(f"Generating {n_samples:,} samples...")
    samples = sampler.sample(n_samples)

    # Get statistics
    sampling_stats = sampler.get_statistics()

    print("\n" + "=" * 60)
    print("SQUEEZE REJECTION SAMPLING RESULTS")
    print("=" * 60)

    print(f"\nTotal proposals made: {sampling_stats['total_proposals']:,}")
    print(f"Total samples generated: {n_samples:,}")
    print(f"π~ evaluations: {sampling_stats['pi_tilde_evaluations']:,}")

    print("\n" + "-" * 40)
    print("PROBABILITY OF NOT EVALUATING π~")
    print("-" * 40)
    print(f"Empirical probability: {sampling_stats['prob_not_evaluating_pi']:.4f}")
    print(f"Theoretical probability: {sampling_stats['theoretical_prob_not_eval']:.4f}")
    print(
        f"Difference: {abs(sampling_stats['prob_not_evaluating_pi'] - sampling_stats['theoretical_prob_not_eval']):.4f}"
    )

    print("\n" + "-" * 40)
    print("ACCEPTANCE RATES")
    print("-" * 40)
    print(f"Step (b) acceptance rate: {sampling_stats['prob_accept_step_b']:.4f}")
    print(f"Step (c) acceptance rate: {sampling_stats['prob_accept_step_c']:.4f}")
    print(f"Overall acceptance rate: {sampling_stats['overall_acceptance_rate']:.4f}")
    print(f"Theoretical acceptance rate: {1 / sampler.M:.4f}")

    if show_plots:
        # Visual verification
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Histogram with theoretical density
        ax1 = axes[0]
        ax1.hist(
            samples,
            bins=50,
            density=True,
            alpha=0.7,
            edgecolor="black",
            label="Samples",
        )
        x_range = np.linspace(-4, 4, 1000)
        ax1.plot(x_range, scipy_stats.norm.pdf(x_range), "r-", lw=2, label="N(0,1)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("Density")
        ax1.set_title("Sample Distribution vs. Standard Normal")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Q-Q plot
        ax2 = axes[1]
        scipy_stats.probplot(samples, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Empirical CDF vs theoretical CDF
        ax3 = axes[2]
        sorted_samples = np.sort(samples)
        empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        theoretical_cdf = scipy_stats.norm.cdf(sorted_samples)
        ax3.plot(sorted_samples, empirical_cdf, "b-", alpha=0.7, label="Empirical CDF")
        ax3.plot(sorted_samples, theoretical_cdf, "r--", lw=2, label="Theoretical CDF")
        ax3.set_xlabel("x")
        ax3.set_ylabel("CDF")
        ax3.set_title("Empirical vs Theoretical CDF")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Statistical tests
    print("\n" + "-" * 40)
    print("STATISTICAL TESTS")
    print("-" * 40)

    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = scipy_stats.kstest(samples, "norm")
    print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4f}")

    # Sample moments
    print(f"\nSample mean: {np.mean(samples):.4f} (theoretical: 0.0000)")
    print(f"Sample std: {np.std(samples, ddof=1):.4f} (theoretical: 1.0000)")
    print(f"Sample skewness: {scipy_stats.skew(samples):.4f} (theoretical: 0.0000)")
    print(f"Sample kurtosis: {scipy_stats.kurtosis(samples):.4f} (theoretical: 0.0000)")

    return samples, sampling_stats


def run_consistency_check(n_runs=10, samples_per_run=10000):
    """Run multiple independent trials to check consistency"""
    print("\n" + "=" * 60)
    print(f"CONSISTENCY CHECK ({n_runs} independent runs)")
    print("=" * 60)

    prob_not_eval_list = []
    acceptance_rates = []

    for i in range(n_runs):
        sampler = SqueezeRejectionSampler()
        _ = sampler.sample(samples_per_run)
        test_stats = sampler.get_statistics()
        prob_not_eval_list.append(test_stats["prob_not_evaluating_pi"])
        acceptance_rates.append(test_stats["overall_acceptance_rate"])
        print(
            f"Run {i + 1:2d}: P(not eval) = {test_stats['prob_not_evaluating_pi']:.4f}, "
            f"Accept rate = {test_stats['overall_acceptance_rate']:.4f}"
        )

    theoretical_prob = 2 * np.sqrt(2) / (3 * np.sqrt(np.e))

    print("\nProbability of not evaluating π~:")
    print(f"  Mean: {np.mean(prob_not_eval_list):.4f}")
    print(f"  Std dev: {np.std(prob_not_eval_list):.4f}")
    print(
        f"  Min: {np.min(prob_not_eval_list):.4f}, Max: {np.max(prob_not_eval_list):.4f}"
    )
    print(f"  Theoretical: {theoretical_prob:.4f}")

    print("\nAcceptance rate:")
    print(f"  Mean: {np.mean(acceptance_rates):.4f}")
    print(f"  Theoretical: {1 / np.sqrt(np.e):.4f}")


# Run the verification
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility

    # Main verification
    samples, sampling_statistics = verify_implementation(
        n_samples=50000, show_plots=True
    )

    # Consistency check
    # run_consistency_check(n_runs=10, samples_per_run=10000)
