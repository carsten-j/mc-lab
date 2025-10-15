import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Create x values
x = np.linspace(-3, 3, 1000)

# Three Gaussians with small variances to create sharp peaks
sigma = 0.05  # Very small standard deviation
component1 = norm.pdf(x, loc=-1, scale=sigma)
component2 = norm.pdf(x, loc=0, scale=sigma)
component3 = norm.pdf(x, loc=1, scale=sigma)

# Equal mixture weights (1/3 each)
mixture_original = (component1 + component2 + component3) / 3

# Different powers for parallel tempering
powers = [1.0, 0.7, 0.5, 0.3, 0.1]
colors = ["#332288", "#117733", "#44AA99", "#CC6677", "#AA4499"]

# Create the plot using project color scheme
plt.figure(figsize=(12, 8))

tempered_distributions = {}

for i, power in enumerate(powers):
    if power == 1.0:
        mixture_tempered_normalized = mixture_original
    else:
        mixture_tempered = mixture_original**power
        # Normalize the tempered distribution to integrate to 1
        normalization_constant = np.trapz(mixture_tempered, x)
        mixture_tempered_normalized = mixture_tempered / normalization_constant

    tempered_distributions[power] = mixture_tempered_normalized

    # Plot each distribution
    if power == 1.0:
        label = f"Original (power={power:.1f}, T={1 / power:.1f})"
    else:
        label = f"Power={power:.1f} (T={1 / power:.1f})"

    plt.plot(x, mixture_tempered_normalized, color=colors[i], linewidth=2, label=label)

plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Parallel Tempering: GMM at Different Temperatures")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-2.5, 2.5)
plt.tight_layout()

# Save the plot
plt.savefig("parallel_tempering_gmm_multiple.png", dpi=300, bbox_inches="tight")
plt.show()

# Print comparison info for all powers
print("Power\tTemp\tPeak Height")
print("-" * 30)
for power in powers:
    peak_height = np.max(tempered_distributions[power])
    temperature = 1 / power
    print(f"{power:.1f}\t{temperature:.1f}\t{peak_height:.2f}")

print("\nAll distributions have area ≈ 1.0")

# Show the effect of tempering on the shape
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x, mixture_original, color="#332288", linewidth=2)
plt.title("Original Distribution (T=1)")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True, alpha=0.3)
plt.xlim(-2.5, 2.5)

plt.subplot(1, 2, 2)
plt.plot(x, mixture_tempered_normalized, color="#CC6677", linewidth=2)
plt.title(f"Tempered Distribution (T={1 / power:.2f})")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True, alpha=0.3)
plt.xlim(-2.5, 2.5)

plt.tight_layout()
plt.savefig("parallel_tempering_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
