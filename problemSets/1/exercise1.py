import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def simulate_truncated_exp_direct(lam, a, size=1000):
    """
    Most efficient method: X = a + Z where Z ~ Exp(λ)
    """
    return a + np.random.exponential(1 / lam, size)


def simulate_truncated_exp_rejection(lam, a, size=1000):
    """
    Rejection sampling: Keep generating until Y ≥ a
    Returns samples and total number of samples generated (including rejected ones)
    """
    samples = []
    total_generated = 0
    while len(samples) < size:
        y = np.random.exponential(1 / lam)
        total_generated += 1
        if y >= a:
            samples.append(y)
    return np.array(samples), total_generated


def simulate_truncated_exp_inverse_transform(lam, a, size=1000):
    """
    Inverse transform method using CDF F_X(x) = 1 - exp(-λ(x-a))
    """
    u = np.random.uniform(0, 1, size)
    return a - np.log(1 - u) / lam


# Example usage
if __name__ == "__main__":
    # Parameters
    lam = 2.0  # rate parameter
    n_samples = 10000

    # Loop over values of a from 1 to 5
    a_values = list(range(1, 6))
    total_generated_values = []

    print("Testing rejection sampling efficiency for different values of a:")
    print("a\tTotal Generated\tAcceptance Rate\tTheoretical Rate")
    print("-" * 60)

    for a in a_values:
        # Generate samples using rejection method
        samples_rejection, total_generated = simulate_truncated_exp_rejection(
            lam, a, n_samples
        )
        total_generated_values.append(total_generated)

        # Calculate acceptance rates
        empirical_rate = n_samples / total_generated
        theoretical_rate = np.exp(-lam * a)

        print(
            f"{a}\t{total_generated}\t\t{empirical_rate:.4f}\t\t{theoretical_rate:.4f}"
        )

    # Create plot showing a against total_generated
    plt.figure(figsize=(10, 6))

    # Plot empirical results
    plt.plot(
        a_values,
        total_generated_values,
        "bo-",
        label="Empirical",
        linewidth=2,
        markersize=8,
    )

    # Plot theoretical expectation: E[total_generated] = n_samples / exp(-λa) = n_samples * exp(λa)
    theoretical_total = [n_samples * np.exp(lam * a) for a in a_values]
    plt.plot(a_values, theoretical_total, "r--", label="Theoretical", linewidth=2)

    plt.xlabel("Truncation Point (a)", fontsize=12)
    plt.ylabel("Total Samples Generated", fontsize=12)
    plt.title(
        f"Rejection Sampling Efficiency vs Truncation Point\n(λ={lam}, n_samples={n_samples})",
        fontsize=14,
    )
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    # plt.yscale("log")  # Log scale since the values grow exponentially
    plt.tight_layout()
    plt.savefig("rejection_sampling_efficiency.pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved plot: rejection_sampling_efficiency.pdf")

    # Also create a plot of acceptance rates
    plt.figure(figsize=(10, 6))

    acceptance_rates = [n_samples / total for total in total_generated_values]
    theoretical_rates = [np.exp(-lam * a) for a in a_values]

    plt.plot(
        a_values, acceptance_rates, "bo-", label="Empirical", linewidth=2, markersize=8
    )
    plt.plot(a_values, theoretical_rates, "r--", label="Theoretical", linewidth=2)

    plt.xlabel("Truncation Point (a)", fontsize=12)
    plt.ylabel("Acceptance Rate", fontsize=12)
    plt.title(
        f"Rejection Sampling Acceptance Rate vs Truncation Point\n(λ={lam})",
        fontsize=14,
    )
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    # plt.yscale("log")  # Log scale since acceptance rates decay exponentially
    plt.tight_layout()
    plt.savefig("rejection_sampling_acceptance_rates.pdf", dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved plot: rejection_sampling_acceptance_rates.pdf")

    # Create comparison histograms for a=2
    a_comparison = 2
    print(f"\nGenerating comparison histograms for a={a_comparison}...")

    # Generate samples using all three methods
    samples_direct = simulate_truncated_exp_direct(lam, a_comparison, n_samples)
    samples_rejection, _ = simulate_truncated_exp_rejection(
        lam, a_comparison, n_samples
    )
    samples_inverse = simulate_truncated_exp_inverse_transform(
        lam, a_comparison, n_samples
    )

    # Set up the figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define common parameters for plotting
    x_range = np.linspace(a_comparison, a_comparison + 5, 1000)
    # PDF of truncated exponential: f(x) = λ * exp(-λ(x-a)) for x ≥ a
    pdf_values = lam * np.exp(-lam * (x_range - a_comparison))

    # Method names and corresponding data
    methods = [
        ("Direct Sampling", samples_direct),
        ("Rejection Sampling", samples_rejection),
        ("Inverse Transform", samples_inverse),
    ]

    # Create histograms with overlaid PDFs
    for i, (method_name, samples) in enumerate(methods):
        # Create histogram using seaborn
        sns.histplot(
            samples,
            bins=50,
            stat="density",
            alpha=0.7,
            color=f"C{i}",
            ax=axes[i],
            label=f"{method_name} (n={len(samples)})",
        )

        # Overlay the theoretical PDF
        axes[i].plot(
            x_range, pdf_values, "r-", linewidth=2, label=f"True PDF (λ={lam})"
        )

        # Customize each subplot
        axes[i].set_xlabel("x", fontsize=12)
        axes[i].set_ylabel("Density", fontsize=12)
        axes[i].set_title(f"{method_name}\n(a={a_comparison}, λ={lam})", fontsize=14)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(a_comparison, a_comparison + 4)

    # Add overall title
    fig.suptitle(
        f"Comparison of Truncated Exponential Sampling Methods (a={a_comparison})",
        fontsize=16,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(
        f"truncated_exp_comparison_a{a_comparison}.pdf", dpi=600, bbox_inches="tight"
    )
    plt.show()
    print(f"Saved comparison plot: truncated_exp_comparison_a{a_comparison}.pdf")


# Additional utility function for single sample generation
def sample_truncated_exp(lam, a):
    """Generate a single sample from truncated exponential"""
    return a + np.random.exponential(1 / lam)
