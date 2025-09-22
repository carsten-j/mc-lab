import matplotlib.pyplot as plt
import numpy as np


class StereographicProjectionSampler2D:
    """
    Stereographic Projection Sampler for 2D distributions
    Maps R^2 to S^2 (2-sphere in 3D) and back
    """

    def __init__(self, R=1.0, h=0.5):
        """
        Parameters:
        R: radius parameter for stereographic projection
        h: step size for random walk on sphere
        """
        self.R = R
        self.h = h
        self.d = 2  # dimension of the target space

    def SP_inverse(self, x):
        """
        Inverse stereographic projection: R^2 -> S^2 \ {North Pole}
        Maps from plane to sphere

        x: point in R^2
        returns: point on S^2 (3D unit sphere)
        """
        x = np.atleast_1d(x).astype(float)
        assert len(x) == 2, "Input must be 2D"

        z = np.zeros(3)

        norm_x_sq = np.sum(x**2)
        denom = norm_x_sq + self.R**2

        # First two components
        z[0] = 2 * self.R * x[0] / denom
        z[1] = 2 * self.R * x[1] / denom
        # Third component (height)
        z[2] = (norm_x_sq - self.R**2) / denom

        # Verify we're on the unit sphere (debugging)
        norm_z = np.linalg.norm(z)
        if abs(norm_z - 1.0) > 1e-10:
            print(f"Warning: ||z|| = {norm_z}, should be 1")
            z = z / norm_z  # Force normalization

        return z

    def SP(self, z):
        """
        Stereographic projection: S^2 \ {North Pole} -> R^2
        Maps from sphere to plane

        z: point on S^2
        returns: point in R^2
        """
        assert len(z) == 3, "Input must be on S^2 (3D)"

        # Ensure z is on unit sphere
        z = z / np.linalg.norm(z)

        x = np.zeros(2)

        denom = 1 - z[2]  # 1 - z_3 (north pole is at (0,0,1))
        if abs(denom) < 1e-10:
            # Handle near north pole
            return np.ones(2) * 1e6  # Large but finite value

        x[0] = self.R * z[0] / denom
        x[1] = self.R * z[1] / denom

        return x

    def propose_on_sphere(self, z):
        """
        Propose new point on sphere using tangent space random walk

        z: current point on S^2
        returns: proposed point on S^2
        """
        # Ensure z is normalized
        z = z / np.linalg.norm(z)

        # Sample random perturbation in R^3
        d_tilde_z = np.random.normal(0, self.h, 3)

        # Project to tangent space at z (perpendicular to z)
        dz = d_tilde_z - np.dot(d_tilde_z, z) * z

        # Add perturbation and re-project to sphere
        z_new = z + dz
        z_new = z_new / np.linalg.norm(z_new)

        return z_new

    def step(self, x_current, log_density_func):
        """
        One step of the SPS algorithm for 2D

        x_current: current state in R^2
        log_density_func: function that computes log density
        returns: (next_state, accepted)
        """
        x_current = np.array(x_current, dtype=float)

        # Map current state to sphere
        z = self.SP_inverse(x_current)

        # Propose new point on sphere
        z_proposed = self.propose_on_sphere(z)

        # Map back to R^2
        x_proposed = self.SP(z_proposed)

        # Check for invalid proposals
        if np.any(np.abs(x_proposed) > 1e5) or np.any(np.isnan(x_proposed)):
            return x_current, False

        # Compute log densities
        try:
            log_dens_current = log_density_func(x_current)
            log_dens_proposed = log_density_func(x_proposed)
        except:
            return x_current, False

        # Compute acceptance probability
        # For d=2: factor is (R^2 + ||x||^2)^2
        norm_x_current_sq = np.sum(x_current**2)
        norm_x_proposed_sq = np.sum(x_proposed**2)

        log_jacobian_ratio = self.d * (
            np.log(self.R**2 + norm_x_proposed_sq)
            - np.log(self.R**2 + norm_x_current_sq)
        )

        log_ratio = log_dens_proposed - log_dens_current + log_jacobian_ratio

        # Clip to avoid overflow
        log_ratio = np.clip(log_ratio, -20, 0)
        accept_prob = min(1.0, np.exp(log_ratio))

        # Accept or reject
        if np.random.rand() < accept_prob:
            return x_proposed, True
        else:
            return x_current, False

    def sample(self, x_init, log_density_func, n_samples, burn_in=1000, verbose=True):
        """
        Generate MCMC samples using SPS

        x_init: initial state (2D vector)
        log_density_func: function that computes log density
        n_samples: number of samples to generate
        burn_in: number of burn-in samples
        verbose: print progress

        returns: (samples, accept_rate)
        """
        samples = np.zeros((n_samples, 2))
        x_current = np.array(x_init, dtype=float)
        n_accepted = 0
        n_accepted_burnin = 0

        # Burn-in
        if verbose:
            print(f"Running burn-in ({burn_in} iterations)...")
        for i in range(burn_in):
            x_current, accepted = self.step(x_current, log_density_func)
            if accepted:
                n_accepted_burnin += 1
            if verbose and (i + 1) % 500 == 0:
                print(
                    f"  Burn-in progress: {i + 1}/{burn_in} (Accept: {n_accepted_burnin / (i + 1):.2%})"
                )

        # Sampling
        if verbose:
            print(f"Sampling ({n_samples} iterations)...")
        for i in range(n_samples):
            x_current, accepted = self.step(x_current, log_density_func)
            samples[i] = x_current
            if accepted:
                n_accepted += 1

            if verbose and (i + 1) % 1000 == 0:
                print(
                    f"  Progress: {i + 1}/{n_samples} (Accept: {n_accepted / (i + 1):.2%})"
                )

        accept_rate = n_accepted / n_samples

        return samples, accept_rate


class RosenbockDistribution:
    """
    Rosenbrock distribution (unnormalized)
    p(x,y) ∝ exp(-((1-x)^2 + 100*(y-x^2)^2) / scale)
    """

    def __init__(self, scale=20):
        self.scale = scale

    def density(self, x):
        """
        Compute unnormalized density at x
        x can be either [x, y] array or separate x, y
        """
        if len(x) == 2:
            x_val, y_val = x[0], x[1]
        else:
            x_val, y_val = x

        return np.exp(-((1 - x_val) ** 2 + 100 * (y_val - x_val**2) ** 2) / self.scale)

    def log_density(self, x):
        """Compute log unnormalized density at x"""
        if len(x) == 2:
            x_val, y_val = x[0], x[1]
        else:
            x_val, y_val = x

        return -((1 - x_val) ** 2 + 100 * (y_val - x_val**2) ** 2) / self.scale


def test_sps_rosenbrock():
    """
    Test SPS on the Rosenbrock distribution and compare with standard RWM
    """
    np.random.seed(42)

    # Create Rosenbrock distribution
    rosenbrock = RosenbockDistribution(scale=20)

    # Standard Random Walk Metropolis for comparison
    class RWMSampler2D:
        def __init__(self, step_size=0.1):
            self.step_size = step_size

        def step(self, x_current, log_density_func):
            x_proposed = x_current + np.random.normal(0, self.step_size, 2)
            log_ratio = log_density_func(x_proposed) - log_density_func(x_current)
            log_ratio = np.clip(log_ratio, -20, 0)
            if np.log(np.random.rand()) < log_ratio:
                return x_proposed, True
            return x_current, False

        def sample(
            self, x_init, log_density_func, n_samples, burn_in=1000, verbose=False
        ):
            samples = np.zeros((n_samples, 2))
            x_current = np.array(x_init, dtype=float)
            n_accepted = 0

            for _ in range(burn_in):
                x_current, _ = self.step(x_current, log_density_func)

            for i in range(n_samples):
                x_current, accepted = self.step(x_current, log_density_func)
                samples[i] = x_current
                if accepted:
                    n_accepted += 1

            return samples, n_accepted / n_samples

    # Run samplers
    n_samples = 10000
    x_init = np.array([-1.0, 0.0])  # Start away from the mode

    # Test different parameter settings for SPS
    print("Testing different SPS parameters...")
    R_values = [1.0, 2.0, 3.0]
    h_values = [0.1, 0.3, 0.5]

    best_accept = 0
    best_R = 1.0
    best_h = 0.3

    for R in R_values:
        for h in h_values:
            test_sampler = StereographicProjectionSampler2D(R=R, h=h)
            test_samples, test_accept = test_sampler.sample(
                x_init, rosenbrock.log_density, 1000, burn_in=500, verbose=False
            )
            print(f"  R={R}, h={h}: Accept rate = {test_accept:.2%}")
            if (
                test_accept > best_accept and test_accept < 0.8
            ):  # Avoid too high acceptance
                best_accept = test_accept
                best_R = R
                best_h = h

    print(f"\nUsing best parameters: R={best_R}, h={best_h}")

    # Run with best parameters
    print("\nRunning Stereographic Projection Sampler...")
    sps_sampler = StereographicProjectionSampler2D(R=best_R, h=best_h)
    sps_samples, sps_accept = sps_sampler.sample(
        x_init, rosenbrock.log_density, n_samples, burn_in=2000, verbose=True
    )

    # RWM for comparison
    print("\nRunning Random Walk Metropolis...")
    rwm_sampler = RWMSampler2D(step_size=0.15)
    rwm_samples, rwm_accept = rwm_sampler.sample(
        x_init, rosenbrock.log_density, n_samples, burn_in=2000
    )

    print("\nFinal Acceptance rates:")
    print(f"SPS: {sps_accept:.2%}")
    print(f"RWM: {rwm_accept:.2%}")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))

    # True density contour
    x_range = np.linspace(-2, 3, 100)
    y_range = np.linspace(-1, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rosenbrock.density([X[i, j], Y[i, j]])

    # 1. SPS samples with contours
    ax1 = plt.subplot(3, 3, 1)
    ax1.contour(X, Y, Z, levels=15, colors="gray", alpha=0.5, linewidths=0.5)
    ax1.scatter(sps_samples[:, 0], sps_samples[:, 1], alpha=0.3, s=1, c="blue")
    ax1.set_title(f"SPS Samples (Accept: {sps_accept:.1%})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim([-2, 3])
    ax1.set_ylim([-1, 4])
    ax1.grid(True, alpha=0.3)

    # 2. RWM samples with contours
    ax2 = plt.subplot(3, 3, 2)
    ax2.contour(X, Y, Z, levels=15, colors="gray", alpha=0.5, linewidths=0.5)
    ax2.scatter(rwm_samples[:, 0], rwm_samples[:, 1], alpha=0.3, s=1, c="red")
    ax2.set_title(f"RWM Samples (Accept: {rwm_accept:.1%})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim([-2, 3])
    ax2.set_ylim([-1, 4])
    ax2.grid(True, alpha=0.3)

    # 3. True density heatmap
    ax3 = plt.subplot(3, 3, 3)
    im = ax3.contourf(X, Y, Z, levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax3)
    ax3.set_title("True Rosenbrock Density")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")

    # 4. SPS trace plot for x coordinate
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(sps_samples[:2000, 0], alpha=0.7, linewidth=0.5)
    ax4.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="True mode")
    ax4.set_title("SPS - X coordinate trace")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("x")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. SPS trace plot for y coordinate
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(sps_samples[:2000, 1], alpha=0.7, linewidth=0.5)
    ax5.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="True mode")
    ax5.set_title("SPS - Y coordinate trace")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("y")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. SPS 2D trajectory
    ax6 = plt.subplot(3, 3, 6)
    ax6.contour(X, Y, Z, levels=10, colors="gray", alpha=0.3, linewidths=0.5)
    trajectory_points = 500
    ax6.plot(
        sps_samples[:trajectory_points, 0],
        sps_samples[:trajectory_points, 1],
        "b-",
        alpha=0.5,
        linewidth=0.5,
    )
    ax6.scatter(
        sps_samples[0, 0],
        sps_samples[0, 1],
        color="green",
        s=50,
        marker="o",
        label="Start",
        zorder=5,
    )
    ax6.scatter(
        sps_samples[trajectory_points - 1, 0],
        sps_samples[trajectory_points - 1, 1],
        color="red",
        s=50,
        marker="s",
        label="End",
        zorder=5,
    )
    ax6.scatter(1, 1, color="yellow", s=100, marker="*", label="True mode", zorder=6)
    ax6.set_title(f"SPS - First {trajectory_points} steps")
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. RWM trace plot for x coordinate
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(rwm_samples[:2000, 0], alpha=0.7, linewidth=0.5, color="red")
    ax7.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="True mode")
    ax7.set_title("RWM - X coordinate trace")
    ax7.set_xlabel("Iteration")
    ax7.set_ylabel("x")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. RWM trace plot for y coordinate
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(rwm_samples[:2000, 1], alpha=0.7, linewidth=0.5, color="red")
    ax8.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="True mode")
    ax8.set_title("RWM - Y coordinate trace")
    ax8.set_xlabel("Iteration")
    ax8.set_ylabel("y")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. RWM 2D trajectory
    ax9 = plt.subplot(3, 3, 9)
    ax9.contour(X, Y, Z, levels=10, colors="gray", alpha=0.3, linewidths=0.5)
    ax9.plot(
        rwm_samples[:trajectory_points, 0],
        rwm_samples[:trajectory_points, 1],
        "r-",
        alpha=0.5,
        linewidth=0.5,
    )
    ax9.scatter(
        rwm_samples[0, 0],
        rwm_samples[0, 1],
        color="green",
        s=50,
        marker="o",
        label="Start",
        zorder=5,
    )
    ax9.scatter(
        rwm_samples[trajectory_points - 1, 0],
        rwm_samples[trajectory_points - 1, 1],
        color="red",
        s=50,
        marker="s",
        label="End",
        zorder=5,
    )
    ax9.scatter(1, 1, color="yellow", s=100, marker="*", label="True mode", zorder=6)
    ax9.set_title(f"RWM - First {trajectory_points} steps")
    ax9.set_xlabel("x")
    ax9.set_ylabel("y")
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.suptitle(
        "Stereographic Projection Sampler vs RWM on Rosenbrock Distribution\n"
        + f"(SPS: R={best_R}, h={best_h})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("sps_vs_rwm_rosenbrock.pdf", dpi=600, bbox_inches="tight")
    plt.show()

    # Additional diagnostics
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Autocorrelation plots

    # X autocorrelation - SPS
    lags = np.arange(100)
    acf_sps_x = [1.0]
    for lag in lags[1:]:
        acf = np.corrcoef(sps_samples[:-lag, 0], sps_samples[lag:, 0])[0, 1]
        acf_sps_x.append(acf)

    axes[0, 0].plot(lags, acf_sps_x, "b-", label="SPS")
    axes[0, 0].axhline(0, color="black", linestyle="-", linewidth=0.5)
    axes[0, 0].set_title("X Autocorrelation")
    axes[0, 0].set_xlabel("Lag")
    axes[0, 0].set_ylabel("ACF")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Y autocorrelation - SPS
    acf_sps_y = [1.0]
    for lag in lags[1:]:
        acf = np.corrcoef(sps_samples[:-lag, 1], sps_samples[lag:, 1])[0, 1]
        acf_sps_y.append(acf)

    axes[0, 1].plot(lags, acf_sps_y, "b-", label="SPS")
    axes[0, 1].axhline(0, color="black", linestyle="-", linewidth=0.5)
    axes[0, 1].set_title("Y Autocorrelation")
    axes[0, 1].set_xlabel("Lag")
    axes[0, 1].set_ylabel("ACF")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Running mean plots
    window = 100
    running_mean_x_sps = np.convolve(
        sps_samples[:, 0], np.ones(window) / window, mode="valid"
    )
    running_mean_y_sps = np.convolve(
        sps_samples[:, 1], np.ones(window) / window, mode="valid"
    )

    axes[0, 2].plot(running_mean_x_sps[:2000], label="X", alpha=0.7)
    axes[0, 2].plot(running_mean_y_sps[:2000], label="Y", alpha=0.7)
    axes[0, 2].axhline(1.0, color="green", linestyle="--", alpha=0.5, label="True mode")
    axes[0, 2].set_title(f"SPS Running Mean (window={window})")
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("Running Mean")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Marginal distributions
    axes[1, 0].hist(
        sps_samples[:, 0], bins=50, alpha=0.5, density=True, color="blue", label="SPS"
    )
    axes[1, 0].hist(
        rwm_samples[:, 0], bins=50, alpha=0.5, density=True, color="red", label="RWM"
    )
    axes[1, 0].axvline(1.0, color="green", linestyle="--", alpha=0.5, label="True mode")
    axes[1, 0].set_title("X Marginal Distribution")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Y marginal
    axes[1, 1].hist(
        sps_samples[:, 1], bins=50, alpha=0.5, density=True, color="blue", label="SPS"
    )
    axes[1, 1].hist(
        rwm_samples[:, 1], bins=50, alpha=0.5, density=True, color="red", label="RWM"
    )
    axes[1, 1].axvline(1.0, color="green", linestyle="--", alpha=0.5, label="True mode")
    axes[1, 1].set_title("Y Marginal Distribution")
    axes[1, 1].set_xlabel("y")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Joint distribution - hexbin
    axes[1, 2].hexbin(sps_samples[:, 0], sps_samples[:, 1], gridsize=30, cmap="Blues")
    axes[1, 2].scatter(
        1, 1, color="red", s=100, marker="*", label="True mode", zorder=5
    )
    axes[1, 2].set_title("SPS Joint Distribution")
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("y")
    axes[1, 2].legend()

    plt.suptitle("Additional Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Calculate summary statistics
    print("\nSummary Statistics:")
    print(
        f"SPS - Mean: x={np.mean(sps_samples[:, 0]):.3f}, y={np.mean(sps_samples[:, 1]):.3f}"
    )
    print(
        f"SPS - Std:  x={np.std(sps_samples[:, 0]):.3f}, y={np.std(sps_samples[:, 1]):.3f}"
    )
    print(
        f"RWM - Mean: x={np.mean(rwm_samples[:, 0]):.3f}, y={np.mean(rwm_samples[:, 1]):.3f}"
    )
    print(
        f"RWM - Std:  x={np.std(rwm_samples[:, 0]):.3f}, y={np.std(rwm_samples[:, 1]):.3f}"
    )

    print("\nTrue mode is at (1, 1)")
    print("Rosenbrock has a narrow curved valley - both samplers should explore it")

    return sps_samples, rwm_samples


if __name__ == "__main__":
    sps_samples, rwm_samples = test_sps_rosenbrock()
