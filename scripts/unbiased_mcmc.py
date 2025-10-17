import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ============================================================================
# PART 1: Define Target Distribution and Kernels
# ============================================================================


class NormalTarget:
    """Target distribution: N(mu, sigma^2)"""

    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def sample_initial(self):
        """Sample from initial distribution (we'll use N(5, 1) to start far from target)"""
        return np.random.normal(5, 1)  # Start far from target

    def log_density(self, x):
        """Log density of target distribution"""
        return -0.5 * ((x - self.mu) / self.sigma) ** 2


# ============================================================================
# PART 2: Define Transition and Coupling Kernels
# ============================================================================


def create_metropolis_kernel(target, proposal_std=1.0):
    """Create a Metropolis-Hastings transition kernel"""

    def transition(x_current):
        # Propose new state
        proposal = x_current + np.random.normal(0, proposal_std)

        # Compute acceptance probability
        log_ratio = target.log_density(proposal) - target.log_density(x_current)
        accept_prob = min(1, np.exp(log_ratio))

        # Accept or reject
        if np.random.rand() < accept_prob:
            return proposal
        else:
            return x_current

    return transition


def create_coupling_kernel(target, proposal_std=1.0):
    """
    Create a coupling kernel using reflection coupling for Gaussian proposals
    """

    def coupling(x, y):
        if np.abs(x - y) < 1e-10:  # Already met (within numerical tolerance)
            # Use common random numbers
            proposal_increment = np.random.normal(0, proposal_std)
            x_proposal = x + proposal_increment

            # Same acceptance for both
            log_ratio = target.log_density(x_proposal) - target.log_density(x)
            accept_prob = min(1, np.exp(log_ratio))

            if np.random.rand() < accept_prob:
                return x_proposal, x_proposal
            else:
                return x, x
        else:
            # Reflection coupling
            # Generate proposal for X
            z = np.random.normal(0, proposal_std)
            x_proposal = x + z

            # Reflect for Y
            midpoint = (x + y) / 2
            if x < y:
                # If z moves x toward y, use same; otherwise reflect
                if z > 0:  # Moving toward y
                    y_proposal = y + z
                else:  # Moving away, reflect
                    distance_to_mid = midpoint - x
                    y_proposal = y - z  # Reflection
            else:  # x > y
                if z < 0:  # Moving toward y
                    y_proposal = y + z
                else:  # Moving away, reflect
                    y_proposal = y - z

            # Independent accept/reject decisions
            # For X
            log_ratio_x = target.log_density(x_proposal) - target.log_density(x)
            accept_prob_x = min(1, np.exp(log_ratio_x))
            if np.random.rand() < accept_prob_x:
                x_new = x_proposal
            else:
                x_new = x

            # For Y
            log_ratio_y = target.log_density(y_proposal) - target.log_density(y)
            accept_prob_y = min(1, np.exp(log_ratio_y))
            if np.random.rand() < accept_prob_y:
                y_new = y_proposal
            else:
                y_new = y

            return x_new, y_new

    return coupling


# ============================================================================
# PART 3: Implement the BasicEstimator Class
# ============================================================================


class BasicEstimator:
    def __init__(self, target_dist, transition_kernel, coupling_kernel):
        self.target = target_dist
        self.transition = transition_kernel
        self.coupling = coupling_kernel
        self.trace_data = []  # Store trace for visualization

    def compute_H_k(self, k, h_function, store_trace=False):
        """
        Compute the basic estimator H_k

        Args:
            k: Fixed time point
            h_function: Function to estimate E[h(X)]
            store_trace: Whether to store chain paths for visualization

        Returns:
            H_k: Unbiased estimate
            tau: Meeting time
            trace: Dictionary with chain paths (if store_trace=True)
        """
        # Initialize chains
        X = [self.target.sample_initial()]
        Y = [self.target.sample_initial()]

        # First step: X moves alone
        X.append(self.transition(X[0]))

        # Coupled evolution
        tau = None
        t = 1

        while tau is None:
            # Couple (X_t, Y_{t-1})
            x_new, y_new = self.coupling(X[t], Y[t - 1])
            X.append(x_new)
            Y.append(y_new)

            # Check meeting (with numerical tolerance)
            if np.abs(x_new - y_new) < 1e-10:
                tau = t + 1
                # Ensure they stay together
                for _ in range(t + 2, max(t + 3, k + 2)):
                    common_value = self.transition(X[-1])
                    X.append(common_value)
                    Y.append(common_value)
                break

            t += 1

            # Safety check
            if t > 10000:
                print(f"Warning: Chains haven't met after {t} iterations")
                tau = t  # Force stop
                break

        # Continue to time k if needed
        while len(X) <= k:
            if tau is not None and len(X) > tau:
                # Use common random numbers after meeting
                new_val = self.transition(X[-1])
                X.append(new_val)
                Y.append(new_val)
            else:
                # Continue coupling
                x_new, y_new = self.coupling(X[-1], Y[-1])
                X.append(x_new)
                Y.append(y_new)

        # Compute estimator
        H_k = h_function(X[k])

        # Add correction terms if tau > k+1
        if tau is not None and tau > k + 1:
            for t in range(k + 1, min(tau, len(X))):
                if t < len(X) and t - 1 < len(Y):
                    H_k += h_function(X[t]) - h_function(Y[t - 1])

        # Store trace if requested
        trace = None
        if store_trace:
            trace = {"X": X, "Y": Y, "tau": tau, "k": k, "H_k": H_k}

        return H_k, tau, trace

    def estimate(self, k, h_function, n_samples=1000, verbose=True):
        """
        Compute estimate using n_samples independent copies
        """
        estimates = []
        meeting_times = []

        if verbose:
            print(f"Running {n_samples} independent simulations...")

        for i in range(n_samples):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{n_samples} simulations")

            H_k, tau, _ = self.compute_H_k(k, h_function, store_trace=False)
            estimates.append(H_k)
            meeting_times.append(tau)

        estimates = np.array(estimates)
        meeting_times = np.array(meeting_times)

        results = {
            "estimate": np.mean(estimates),
            "std_error": np.std(estimates) / np.sqrt(n_samples),
            "variance": np.var(estimates),
            "min": np.min(estimates),
            "max": np.max(estimates),
            "avg_meeting_time": np.mean(meeting_times),
            "median_meeting_time": np.median(meeting_times),
            "max_meeting_time": np.max(meeting_times),
            "all_estimates": estimates,
            "all_meeting_times": meeting_times,
        }

        if verbose:
            print("\n" + "=" * 50)
            print("RESULTS")
            print("=" * 50)
            print(f"Estimate: {results['estimate']:.6f}")
            print(f"Standard Error: {results['std_error']:.6f}")
            print(
                f"95% CI: [{results['estimate'] - 1.96 * results['std_error']:.6f}, "
                f"{results['estimate'] + 1.96 * results['std_error']:.6f}]"
            )
            print(f"Average Meeting Time: {results['avg_meeting_time']:.1f}")
            print(f"Median Meeting Time: {results['median_meeting_time']:.1f}")
            print(f"Max Meeting Time: {results['max_meeting_time']:.0f}")

        return results


# ============================================================================
# PART 4: Example Usage
# ============================================================================


def example_basic_usage():
    """Basic example: Estimate mean of N(0,1)"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Estimating E[X] for X ~ N(0,1)")
    print("=" * 60)

    # Setup
    target = NormalTarget(mu=0, sigma=1)
    transition = create_metropolis_kernel(target, proposal_std=1.5)
    coupling = create_coupling_kernel(target, proposal_std=1.5)

    # Create estimator
    estimator = BasicEstimator(target, transition, coupling)

    # Define function h (we want E[X], so h(x) = x)
    h_function = lambda x: x

    # Run estimation for different values of k
    k_values = [5, 10, 20]

    for k in k_values:
        print(f"\n--- Using k = {k} ---")
        results = estimator.estimate(k, h_function, n_samples=500, verbose=False)
        print(f"Estimate: {results['estimate']:.6f}")
        print(f"Standard Error: {results['std_error']:.6f}")
        print(f"Average Meeting Time: {results['avg_meeting_time']:.1f}")

    print("\nTrue value: 0.000000")


def example_with_visualization():
    """Example with visualization of chain paths"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Visualizing Chain Coupling")
    print("=" * 60)

    # Setup
    target = NormalTarget(mu=0, sigma=1)
    transition = create_metropolis_kernel(target, proposal_std=1.5)
    coupling = create_coupling_kernel(target, proposal_std=1.5)

    # Create estimator
    estimator = BasicEstimator(target, transition, coupling)

    # Get a single trace for visualization
    k = 10
    h_function = lambda x: x
    H_k, tau, trace = estimator.compute_H_k(k, h_function, store_trace=True)

    print("Single run results:")
    print(f"  H_k = {H_k:.6f}")
    print(f"  Meeting time τ = {tau}")
    print(f"  Value at time k: X_k = {trace['X'][k]:.6f}")

    # Plot the chains
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot chain paths
    t_max = len(trace["X"])
    t_range = range(t_max)

    ax1.plot(t_range, trace["X"], "b-", label="X chain", alpha=0.7, linewidth=2)
    ax1.plot(
        range(len(trace["Y"])),
        trace["Y"],
        "r--",
        label="Y chain (lagged)",
        alpha=0.7,
        linewidth=2,
    )
    ax1.axvline(x=k, color="green", linestyle=":", label=f"k={k}", linewidth=2)
    if tau is not None and tau < t_max:
        ax1.axvline(x=tau, color="purple", linestyle=":", label=f"τ={tau}", linewidth=2)
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Chain Value")
    ax1.set_title("Chain Paths")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot differences
    differences = []
    for t in range(1, min(len(trace["X"]), len(trace["Y"]) + 1)):
        if t <= len(trace["Y"]):
            differences.append(trace["X"][t] - trace["Y"][t - 1])

    ax2.plot(range(1, len(differences) + 1), differences, "g-", linewidth=2)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax2.axvline(x=k, color="green", linestyle=":", label=f"k={k}", linewidth=2)
    if tau is not None and tau < len(differences) + 1:
        ax2.axvline(x=tau, color="purple", linestyle=":", label=f"τ={tau}", linewidth=2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("X_t - Y_{t-1}")
    ax2.set_title("Chain Differences (become 0 after meeting)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def example_different_functions():
    """Example: Estimating different functions of the target"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Estimating Different Functions")
    print("=" * 60)

    # Setup
    target = NormalTarget(mu=0, sigma=1)
    transition = create_metropolis_kernel(target, proposal_std=1.5)
    coupling = create_coupling_kernel(target, proposal_std=1.5)

    # Create estimator
    estimator = BasicEstimator(target, transition, coupling)

    # Different functions to estimate
    functions = {
        "E[X]": (lambda x: x, 0.0),  # (function, true_value)
        "E[X²]": (lambda x: x**2, 1.0),
        "E[X³]": (lambda x: x**3, 0.0),
        "E[|X|]": (lambda x: abs(x), np.sqrt(2 / np.pi)),
        "E[exp(X)]": (lambda x: np.exp(x), np.exp(0.5)),
        "P(X > 0)": (lambda x: 1.0 if x > 0 else 0.0, 0.5),
    }

    k = 10
    n_samples = 1000

    print(f"\nUsing k={k}, n_samples={n_samples}")
    print("-" * 60)
    print(
        f"{'Function':<15} {'Estimate':<12} {'True Value':<12} {'Error':<12} {'Std Error':<12}"
    )
    print("-" * 60)

    for name, (h_func, true_val) in functions.items():
        results = estimator.estimate(k, h_func, n_samples=n_samples, verbose=False)
        error = results["estimate"] - true_val
        print(
            f"{name:<15} {results['estimate']:<12.6f} {true_val:<12.6f} "
            f"{error:<12.6f} {results['std_error']:<12.6f}"
        )


def example_meeting_time_analysis():
    """Analyze how meeting times depend on k and proposal variance"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Meeting Time Analysis")
    print("=" * 60)

    target = NormalTarget(mu=0, sigma=1)

    # Test different proposal standard deviations
    proposal_stds = [0.5, 1.0, 1.5, 2.0, 2.5]
    k_values = [5, 10, 20]

    results_matrix = {}

    for prop_std in proposal_stds:
        transition = create_metropolis_kernel(target, proposal_std=prop_std)
        coupling = create_coupling_kernel(target, proposal_std=prop_std)
        estimator = BasicEstimator(target, transition, coupling)

        for k in k_values:
            # Collect meeting times
            meeting_times = []
            for _ in range(100):  # Smaller sample for speed
                _, tau, _ = estimator.compute_H_k(k, lambda x: x, store_trace=False)
                meeting_times.append(tau)

            avg_tau = np.mean(meeting_times)
            results_matrix[(prop_std, k)] = avg_tau

    # Display results
    print("\nAverage Meeting Times:")
    print("-" * 60)
    print(f"{'Proposal Std':<15}", end="")
    for k in k_values:
        print(f"k={k:<8}", end="")
    print()
    print("-" * 60)

    for prop_std in proposal_stds:
        print(f"{prop_std:<15.1f}", end="")
        for k in k_values:
            avg_tau = results_matrix[(prop_std, k)]
            print(f"{avg_tau:<10.1f}", end="")
        print()

    print(
        "\nNote: Meeting time is independent of k, but we need to run until max(k, τ)"
    )


def example_convergence_diagnostic():
    """Use basic estimator as a convergence diagnostic"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Convergence Diagnostic")
    print("=" * 60)

    # Setup with intentionally poor starting point
    target = NormalTarget(mu=0, sigma=1)
    transition = create_metropolis_kernel(target, proposal_std=1.5)
    coupling = create_coupling_kernel(target, proposal_std=1.5)

    # Test different k values to see convergence
    k_values = list(range(1, 31, 2))
    n_samples = 200

    estimates = []
    std_errors = []
    avg_meeting_times = []

    print("Testing convergence as k increases...")

    for k in k_values:
        estimator = BasicEstimator(target, transition, coupling)
        results = estimator.estimate(k, lambda x: x, n_samples=n_samples, verbose=False)
        estimates.append(results["estimate"])
        std_errors.append(results["std_error"])
        avg_meeting_times.append(results["avg_meeting_time"])

    # Plot convergence
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # Estimates with error bars
    ax1.errorbar(
        k_values,
        estimates,
        yerr=[1.96 * se for se in std_errors],
        fmt="o-",
        capsize=5,
        label="Estimate ± 95% CI",
    )
    ax1.axhline(y=0, color="red", linestyle="--", label="True value")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Estimate of E[X]")
    ax1.set_title("Convergence of Basic Estimator")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Standard errors
    ax2.plot(k_values, std_errors, "o-", color="orange")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Standard Error")
    ax2.set_title("Standard Error vs k")
    ax2.grid(True, alpha=0.3)

    # Meeting times
    ax3.plot(k_values, avg_meeting_times, "o-", color="green")
    ax3.set_xlabel("k")
    ax3.set_ylabel("Average Meeting Time")
    ax3.set_title("Average Meeting Time (should be independent of k)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def example_distribution_comparison():
    """Compare the distribution of estimates for different k"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Distribution of Estimates")
    print("=" * 60)

    target = NormalTarget(mu=0, sigma=1)
    transition = create_metropolis_kernel(target, proposal_std=1.5)
    coupling = create_coupling_kernel(target, proposal_std=1.5)

    k_values = [5, 10, 20]
    n_samples = 1000

    all_estimates = {}

    for k in k_values:
        print(f"Generating estimates for k={k}...")
        estimator = BasicEstimator(target, transition, coupling)
        results = estimator.estimate(k, lambda x: x, n_samples=n_samples, verbose=False)
        all_estimates[k] = results["all_estimates"]

    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, k in enumerate(k_values):
        ax = axes[idx]

        # Histogram
        ax.hist(
            all_estimates[k],
            bins=30,
            density=True,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )

        # Overlay normal distribution with same mean and std
        mean = np.mean(all_estimates[k])
        std = np.std(all_estimates[k])
        x_range = np.linspace(mean - 4 * std, mean + 4 * std, 100)
        ax.plot(
            x_range,
            stats.norm.pdf(x_range, mean, std),
            "r-",
            linewidth=2,
            label="Normal fit",
        )

        # Add vertical line at true value
        ax.axvline(x=0, color="green", linestyle="--", linewidth=2, label="True value")

        ax.set_title(f"k={k}\nMean={mean:.4f}, Std={std:.4f}")
        ax.set_xlabel("Estimate")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Distribution of Basic Estimator for Different k", fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================================
# PART 5: Main Execution
# ============================================================================

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run all examples
    example_basic_usage()
    # example_different_functions()
    # example_meeting_time_analysis()
    # example_with_visualization()
    # example_convergence_diagnostic()
    # example_distribution_comparison()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    Key Observations:
    1. The basic estimator H_k is unbiased for any k ≥ 0
    2. Meeting time τ is independent of k (depends on coupling quality)
    3. Larger k generally gives estimates closer to stationarity
    4. Standard error depends on both k and the coupling efficiency
    5. The estimator works for any function h, not just the mean
    
    Practical Tips:
    - Choose k large enough for burn-in but not wastefully large
    - Better coupling (appropriate proposal variance) reduces meeting times
    - Monitor meeting times - if too large, coupling may need improvement
    - Use multiple independent runs to get reliable estimates
    """)
