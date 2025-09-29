"""
Exercise 1: Gibbs Sampler for Problematic Distribution

This script demonstrates the conditional distributions from Problem Set 2, Exercise 1:
π(x,y) ∝ exp(-½(x-1)²(y-2)²)

The conditional distributions are:
- π(x|y) = N(1, 1/(y-2)²) for y ≠ 2
- π(y|x) = N(2, 1/(x-1)²) for x ≠ 1

This example shows why the Gibbs sampler fails for this distribution.
"""

import warnings
from typing import Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from mc_lab._rng import RandomState, as_generator
from mc_lab.gibbs_sampler_2d import GibbsSampler2D


def sample_x_given_y(y: float, rng: Optional[RandomState] = None) -> float:
    """
    Sample from π(x|y) = N(1, 1/(y-2)²) for y ≠ 2.

    Parameters:
    -----------
    y : float
        Current value of y
    rng : RandomState, optional
        Random number generator

    Returns:
    --------
    float
        Sample from conditional distribution
    """
    generator = as_generator(rng)

    # Check for problematic case
    if abs(y - 2.0) < 1e-6:
        warnings.warn(
            f"y={y} is very close to 2, conditional variance is extremely large"
        )
        # Use a large but finite variance to avoid complete failure
        variance = 1e6
    else:
        variance = 1.0 / ((y - 2.0) ** 2)

    # Cap variance to prevent numerical issues
    variance = min(variance, 1e6)

    return generator.normal(loc=1.0, scale=np.sqrt(variance))


def sample_y_given_x(x: float, rng: Optional[RandomState] = None) -> float:
    """
    Sample from π(y|x) = N(2, 1/(x-1)²) for x ≠ 1.

    Parameters:
    -----------
    x : float
        Current value of x
    rng : RandomState, optional
        Random number generator

    Returns:
    --------
    float
        Sample from conditional distribution
    """
    generator = as_generator(rng)

    # Check for problematic case
    if abs(x - 1.0) < 1e-6:
        warnings.warn(
            f"x={x} is very close to 1, conditional variance is extremely large"
        )
        # Use a large but finite variance to avoid complete failure
        variance = 1e6
    else:
        variance = 1.0 / ((x - 1.0) ** 2)

    # Cap variance to prevent numerical issues
    variance = min(variance, 1e6)

    return generator.normal(loc=2.0, scale=np.sqrt(variance))


def log_prob(x: float, y: float) -> float:
    """
    Compute log π(x,y) = -½(x-1)²(y-2)²

    Parameters:
    -----------
    x, y : float
        Point to evaluate

    Returns:
    --------
    float
        Log probability
    """
    return -0.5 * (x - 1.0) ** 2 * (y - 2.0) ** 2


def demonstrate_problematic_gibbs():
    """
    Demonstrate why the Gibbs sampler fails for this distribution.
    """
    print("=== Exercise 1: Problematic Gibbs Sampler ===\n")

    print("Target distribution: π(x,y) ∝ exp(-½(x-1)²(y-2)²)")
    print("\nConditional distributions:")
    print("- π(x|y) = N(1, 1/(y-2)²) for y ≠ 2")
    print("- π(y|x) = N(2, 1/(x-1)²) for x ≠ 1")
    print("\nPROBLEM: The conditional variances become infinite when y=2 or x=1")
    print("This makes the distribution improper and the Gibbs sampler unstable.\n")

    # Create Gibbs sampler with wrapped functions (no rng parameter)
    def sample_x_wrapper(y):
        return sample_x_given_y(y, rng=42)  # Fixed seed for reproducibility

    def sample_y_wrapper(x):
        return sample_y_given_x(x, rng=42)  # Fixed seed for reproducibility

    sampler = GibbsSampler2D(
        sample_x_given_y=sample_x_wrapper,
        sample_y_given_x=sample_y_wrapper,
        log_prob=log_prob,
        var_names=("x", "y"),
    )

    print("Attempting to run Gibbs sampler with safe initial values...")

    try:
        # Use initial values away from the problematic lines
        initial_state = np.array([0.0, 1.0])  # Away from x=1, y=2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            idata = sampler.sample(
                n_samples=2000,
                n_chains=4,
                burn_in=500,
                initial_state=initial_state,
                random_seed=42,
                progressbar=True,
            )

            # Print any warnings that occurred
            if w:
                print(f"\nWarnings during sampling ({len(w)} total):")
                for warning in w[:3]:  # Show first 3 warnings
                    print(f"  - {warning.message}")
                if len(w) > 3:
                    print(f"  ... and {len(w) - 3} more warnings")

        print("\nSampling completed (somehow)!")
        print(f"Final samples shape: {idata.posterior.x.shape}")

        # Show some statistics
        x_samples = idata.posterior.x.values.flatten()
        y_samples = idata.posterior.y.values.flatten()

        print("\nSample statistics:")
        print(f"X: mean={np.mean(x_samples):.3f}, std={np.std(x_samples):.3f}")
        print(f"Y: mean={np.mean(y_samples):.3f}, std={np.std(y_samples):.3f}")
        print(f"X range: [{np.min(x_samples):.3f}, {np.max(x_samples):.3f}]")
        print(f"Y range: [{np.min(y_samples):.3f}, {np.max(y_samples):.3f}]")

        # Check if we got close to problematic values
        close_to_x1 = np.sum(np.abs(x_samples - 1.0) < 0.1)
        close_to_y2 = np.sum(np.abs(y_samples - 2.0) < 0.1)

        print("\nSamples close to problematic values:")
        print(f"Close to x=1 (within 0.1): {close_to_x1}/{len(x_samples)}")
        print(f"Close to y=2 (within 0.1): {close_to_y2}/{len(y_samples)}")

        return idata

    except Exception as e:
        print(f"\nSampling failed with error: {e}")
        print(
            "This demonstrates why the Gibbs sampler doesn't work for this distribution."
        )
        return None


def plot_conditional_variance_behavior():
    """
    Plot how the conditional variances behave near the problematic values.
    """
    print("\n=== Conditional Variance Behavior ===")

    # Plot variance of X|Y as function of y
    y_vals = np.linspace(1.5, 2.5, 1000)
    y_vals = y_vals[y_vals != 2.0]  # Remove exact problematic point

    var_x_given_y = 1.0 / (y_vals - 2.0) ** 2

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(y_vals, var_x_given_y, "b-", linewidth=2)
    plt.axvline(x=2.0, color="r", linestyle="--", label="y=2 (problematic)")
    plt.yscale("log")
    plt.xlabel("y")
    plt.ylabel("Var(X|y)")
    plt.title("Conditional Variance of X given Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(1e0, 1e6)

    # Plot variance of Y|X as function of x
    x_vals = np.linspace(0.5, 1.5, 1000)
    x_vals = x_vals[x_vals != 1.0]  # Remove exact problematic point

    var_y_given_x = 1.0 / (x_vals - 1.0) ** 2

    plt.subplot(1, 2, 2)
    plt.plot(x_vals, var_y_given_x, "g-", linewidth=2)
    plt.axvline(x=1.0, color="r", linestyle="--", label="x=1 (problematic)")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("Var(Y|x)")
    plt.title("Conditional Variance of Y given X")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(1e0, 1e6)

    plt.tight_layout()
    plt.savefig(
        "/Users/carsten/Dev/mc-lab/problemSets/2/exercise1_conditional_variances.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    print("Conditional variance plot saved as exercise1_conditional_variances.png")
    print("\nObservation: Variances explode to infinity as y→2 or x→1")


if __name__ == "__main__":
    # Demonstrate the problematic Gibbs sampler
    idata = demonstrate_problematic_gibbs()

    az.plot_trace(idata)

    # Plot the conditional variance behavior
    plot_conditional_variance_behavior()

    print("\n=== CONCLUSION ===")
    print("This distribution is IMPROPER and the Gibbs sampler FAILS because:")
    print("1. Conditional variances become infinite at x=1 and y=2")
    print("2. The joint distribution doesn't integrate to finite value")
    print("3. The sampler exhibits unstable behavior near modal lines")
    print("\nThis example illustrates why proper distributions are required for MCMC!")
