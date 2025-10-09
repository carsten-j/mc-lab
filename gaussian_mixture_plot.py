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
mixture = (component1 + component2 + component3) / 3

# Create the plot using project color scheme
plt.figure(figsize=(10, 6))
plt.plot(x, mixture, color="#CC6677", linewidth=2, label="Gaussian Mixture Model")

# Also plot individual components for reference
plt.plot(
    x,
    component1 / 3,
    "--",
    color="#332288",
    alpha=0.7,
    linewidth=1,
    label="Component 1 (μ=-1)",
)
plt.plot(
    x,
    component2 / 3,
    "--",
    color="#117733",
    alpha=0.7,
    linewidth=1,
    label="Component 2 (μ=0)",
)
plt.plot(
    x,
    component3 / 3,
    "--",
    color="#44AA99",
    alpha=0.7,
    linewidth=1,
    label="Component 3 (μ=1)",
)

plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Gaussian Mixture Model with Three Components (σ=0.05)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-2.5, 2.5)
plt.tight_layout()

# Save the plot
plt.savefig("gaussian_mixture_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# Print some info
print(f"Peak height of each component: {np.max(component1 / 3):.2f}")
print(f"Total area under mixture: {np.trapz(mixture, x):.3f}")
