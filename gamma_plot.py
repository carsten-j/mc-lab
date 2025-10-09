import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

# Create x values
x = np.linspace(0, 8, 1000)

# Gamma distribution with shape parameter gamma=1 (equivalent to exponential with rate=1)
gamma_dist = gamma(a=1, scale=1)  # shape=1, scale=1
y = gamma_dist.pdf(x)

# Print some debug info
print(f"First few x values: {x[:5]}")
print(f"First few y values: {y[:5]}")
print(f"Max y value: {y.max()}")
print(f"Distribution mean: {gamma_dist.mean()}")

# Create the plot using project color scheme
plt.figure(figsize=(10, 6))
plt.plot(x, y, color="#332288", linewidth=2, label="Gamma(γ=1)")
plt.fill_between(x, y, alpha=0.3, color="#332288")

plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Gamma Distribution with γ=1 (Exponential Distribution)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 8)
plt.ylim(0, 1.1)
plt.tight_layout()

# Save the plot
plt.savefig("gamma_distribution_plot.png", dpi=300, bbox_inches="tight")
plt.show()
