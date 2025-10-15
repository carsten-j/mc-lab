import matplotlib.pyplot as plt
import numpy as np


def U(x, gamma):
    """Potential function U(x) = gamma * (x^2 - 1)^2"""
    return gamma * (x**2 - 1) ** 2


def target(x, gamma):
    """target distribution"""
    return np.exp(-U(x, gamma))


# Create plots for gamma = 4
gamma = 4
x = np.linspace(-2, 2, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Potential function U(x)
ax1.plot(x, U(x, gamma), color="#332288", linewidth=2)
ax1.set_xlabel("x")
ax1.set_ylabel("U(x)")
ax1.set_title(f"Potential Function U(x) for γ = {gamma}")
ax1.grid(True, alpha=0.3)

# Plot 2: Target distribution
ax2.plot(x, target(x, gamma), color="#117733", linewidth=2)
ax2.set_xlabel("x")
ax2.set_ylabel("π(x)")
ax2.set_title(f"Target Distribution π(x) = exp(-U(x)) for γ = {gamma}")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
