#!/usr/bin/env python3
"""
Optimized Stereographic Projection Sampler with Numba JIT compilation

Performance improvements:
1. Numba JIT compilation of critical path methods
2. Pure NumPy bimodal distribution (avoiding SciPy overhead)
3. Reduced array allocations in tight loops
4. Vectorized RNG calls
"""

import timeit

import matplotlib.pyplot as plt
import numpy as np

# Optional Numba acceleration
try:
    import numba as nb

    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    nb = None


class OptimizedStereographicProjectionSampler1D:
    """
    Numba-optimized Stereographic Projection Sampler for 1D distributions
    """

    def __init__(self, R=1.0, h=0.5):
        """
        Parameters:
        R: radius parameter for stereographic projection
        h: step size for random walk on sphere
        """
        self.R = R
        self.h = h
        self.R_sq = R * R

    def sample(self, x_init, log_density_func, n_samples, burn_in=1000):
        """
        Generate MCMC samples using optimized SPS
        """
        if _HAVE_NUMBA:
            return self._sample_numba(x_init, log_density_func, n_samples, burn_in)
        else:
            # Fallback to original implementation
            return self._sample_python(x_init, log_density_func, n_samples, burn_in)

    def _sample_python(self, x_init, log_density_func, n_samples, burn_in):
        """Python fallback implementation"""
        samples = np.zeros(n_samples)
        x_current = float(x_init)
        n_accepted = 0

        # Burn-in
        for _ in range(burn_in):
            x_current, _ = self._step_python(x_current, log_density_func)

        # Sampling
        for i in range(n_samples):
            x_current, accepted = self._step_python(x_current, log_density_func)
            samples[i] = x_current
            if accepted:
                n_accepted += 1

        accept_rate = n_accepted / n_samples
        return samples, accept_rate

    def _step_python(self, x_current, log_density_func):
        """Python implementation of single step"""
        # Map current state to circle
        z = self._SP_inverse_python(x_current)

        # Propose new point on circle
        z_proposed = self._propose_on_sphere_python(z)

        # Map back to R^1
        x_proposed = self._SP_python(z_proposed)

        # Check for invalid proposals
        if np.isinf(x_proposed) or np.isnan(x_proposed):
            return x_current, False

        # Compute acceptance probability
        x_current_sq = x_current * x_current
        x_proposed_sq = x_proposed * x_proposed

        log_ratio = (
            log_density_func(x_proposed)
            - log_density_func(x_current)
            + np.log(self.R_sq + x_proposed_sq)
            - np.log(self.R_sq + x_current_sq)
        )

        # Accept/reject
        if log_ratio >= 0:
            return x_proposed, True
        elif np.random.rand() < np.exp(log_ratio):
            return x_proposed, True
        else:
            return x_current, False

    def _SP_inverse_python(self, x):
        """Python implementation of inverse stereographic projection"""
        x_sq = x * x
        denom_inv = 1.0 / (x_sq + self.R_sq)
        z0 = 2 * self.R * x * denom_inv
        z1 = (x_sq - self.R_sq) * denom_inv
        return np.array([z0, z1])

    def _SP_python(self, z):
        """Python implementation of stereographic projection"""
        denom = 1 - z[1]
        if abs(denom) < 1e-10:
            return np.inf
        return self.R * z[0] / denom

    def _propose_on_sphere_python(self, z):
        """Python implementation of sphere proposal"""
        d_tilde_z0 = np.random.normal(0, self.h)
        d_tilde_z1 = np.random.normal(0, self.h)

        z_dot_d = z[0] * d_tilde_z0 + z[1] * d_tilde_z1

        dz0 = d_tilde_z0 - z_dot_d * z[0]
        dz1 = d_tilde_z1 - z_dot_d * z[1]

        z_new0 = z[0] + dz0
        z_new1 = z[1] + dz1

        norm_inv = 1.0 / np.sqrt(z_new0 * z_new0 + z_new1 * z_new1)
        z_new0 *= norm_inv
        z_new1 *= norm_inv

        return np.array([z_new0, z_new1])

    if _HAVE_NUMBA:

        def _sample_numba(self, x_init, log_density_func, n_samples, burn_in):
            """Numba-optimized sampling"""
            # We need to handle the log_density_func specially since Numba can't JIT arbitrary Python functions
            # For now, assume it's a BimodalDistribution and extract parameters
            if hasattr(log_density_func, "__self__") and hasattr(
                log_density_func.__self__, "mu1"
            ):
                bimodal = log_density_func.__self__
                return _sample_sps_numba(
                    x_init,
                    n_samples,
                    burn_in,
                    self.R,
                    self.h,
                    bimodal.mu1,
                    bimodal.sigma1,
                    bimodal.mu2,
                    bimodal.sigma2,
                    bimodal.weight,
                )
            else:
                # Fallback to Python implementation for general functions
                return self._sample_python(x_init, log_density_func, n_samples, burn_in)


class OptimizedBimodalDistribution:
    """
    Optimized bimodal distribution using direct NumPy calculations
    """

    def __init__(self, mu1=-3, sigma1=1, mu2=3, sigma2=1, weight=0.5):
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.weight = weight

        # Precompute constants for performance
        self.sigma1_sq = sigma1 * sigma1
        self.sigma2_sq = sigma2 * sigma2
        self.log_sigma1 = np.log(sigma1)
        self.log_sigma2 = np.log(sigma2)
        self.inv_sigma1_sq = 1.0 / self.sigma1_sq
        self.inv_sigma2_sq = 1.0 / self.sigma2_sq
        self.log_weight = np.log(weight)
        self.log_one_minus_weight = np.log(1 - weight)

    def density(self, x):
        """Fast density computation"""
        # Avoid scipy.stats.norm.pdf overhead
        diff1 = x - self.mu1
        diff2 = x - self.mu2

        exp1 = np.exp(-0.5 * diff1 * diff1 * self.inv_sigma1_sq)
        exp2 = np.exp(-0.5 * diff2 * diff2 * self.inv_sigma2_sq)

        const1 = 1.0 / (self.sigma1 * np.sqrt(2 * np.pi))
        const2 = 1.0 / (self.sigma2 * np.sqrt(2 * np.pi))

        comp1 = self.weight * const1 * exp1
        comp2 = (1 - self.weight) * const2 * exp2

        return comp1 + comp2

    def log_density(self, x):
        """Fast log density computation using log-sum-exp trick"""
        diff1 = x - self.mu1
        diff2 = x - self.mu2

        # Log of unnormalized densities
        log_unnorm1 = (
            self.log_weight - self.log_sigma1 - 0.5 * diff1 * diff1 * self.inv_sigma1_sq
        )
        log_unnorm2 = (
            self.log_one_minus_weight
            - self.log_sigma2
            - 0.5 * diff2 * diff2 * self.inv_sigma2_sq
        )

        # Log-sum-exp trick for numerical stability
        max_log = max(log_unnorm1, log_unnorm2)
        log_sum = max_log + np.log(
            np.exp(log_unnorm1 - max_log) + np.exp(log_unnorm2 - max_log)
        )

        # Subtract log(sqrt(2*pi))
        return log_sum - 0.5 * np.log(2 * np.pi)


if _HAVE_NUMBA:

    @nb.njit(cache=True, fastmath=True)
    def _bimodal_log_density_numba(x, mu1, sigma1, mu2, sigma2, weight):
        """Numba-compiled bimodal log density"""
        sigma1_sq = sigma1 * sigma1
        sigma2_sq = sigma2 * sigma2
        inv_sigma1_sq = 1.0 / sigma1_sq
        inv_sigma2_sq = 1.0 / sigma2_sq

        diff1 = x - mu1
        diff2 = x - mu2

        log_unnorm1 = (
            np.log(weight) - np.log(sigma1) - 0.5 * diff1 * diff1 * inv_sigma1_sq
        )
        log_unnorm2 = (
            np.log(1 - weight) - np.log(sigma2) - 0.5 * diff2 * diff2 * inv_sigma2_sq
        )

        # Log-sum-exp
        max_log = max(log_unnorm1, log_unnorm2)
        log_sum = max_log + np.log(
            np.exp(log_unnorm1 - max_log) + np.exp(log_unnorm2 - max_log)
        )

        return log_sum - 0.5 * np.log(2 * np.pi)

    @nb.njit(cache=True, fastmath=True)
    def _SP_inverse_numba(x, R, R_sq):
        """Numba-compiled inverse stereographic projection"""
        x_sq = x * x
        denom_inv = 1.0 / (x_sq + R_sq)
        z0 = 2 * R * x * denom_inv
        z1 = (x_sq - R_sq) * denom_inv
        return z0, z1

    @nb.njit(cache=True, fastmath=True)
    def _SP_numba(z0, z1, R):
        """Numba-compiled stereographic projection"""
        denom = 1 - z1
        if abs(denom) < 1e-10:
            return np.inf
        return R * z0 / denom

    @nb.njit(cache=True, fastmath=True)
    def _propose_on_sphere_numba(z0, z1, h):
        """Numba-compiled sphere proposal"""
        d_tilde_z0 = np.random.normal(0, h)
        d_tilde_z1 = np.random.normal(0, h)

        z_dot_d = z0 * d_tilde_z0 + z1 * d_tilde_z1

        dz0 = d_tilde_z0 - z_dot_d * z0
        dz1 = d_tilde_z1 - z_dot_d * z1

        z_new0 = z0 + dz0
        z_new1 = z1 + dz1

        norm_inv = 1.0 / np.sqrt(z_new0 * z_new0 + z_new1 * z_new1)
        z_new0 *= norm_inv
        z_new1 *= norm_inv

        return z_new0, z_new1

    @nb.njit(cache=True, fastmath=True)
    def _step_numba(x_current, R, h, mu1, sigma1, mu2, sigma2, weight):
        """Numba-compiled single step"""
        R_sq = R * R

        # Map to circle
        z0, z1 = _SP_inverse_numba(x_current, R, R_sq)

        # Propose on sphere
        z_prop0, z_prop1 = _propose_on_sphere_numba(z0, z1, h)

        # Map back
        x_proposed = _SP_numba(z_prop0, z_prop1, R)

        # Check validity
        if np.isinf(x_proposed) or np.isnan(x_proposed):
            return x_current, False

        # Acceptance probability
        x_current_sq = x_current * x_current
        x_proposed_sq = x_proposed * x_proposed

        log_ratio = (
            _bimodal_log_density_numba(x_proposed, mu1, sigma1, mu2, sigma2, weight)
            - _bimodal_log_density_numba(x_current, mu1, sigma1, mu2, sigma2, weight)
            + np.log(R_sq + x_proposed_sq)
            - np.log(R_sq + x_current_sq)
        )

        # Accept/reject
        if log_ratio >= 0:
            return x_proposed, True
        elif np.random.random() < np.exp(log_ratio):
            return x_proposed, True
        else:
            return x_current, False

    @nb.njit(cache=True, fastmath=True)
    def _sample_sps_numba(
        x_init, n_samples, burn_in, R, h, mu1, sigma1, mu2, sigma2, weight
    ):
        """Numba-compiled full sampling loop"""
        samples = np.zeros(n_samples)
        x_current = x_init
        n_accepted = 0

        # Burn-in
        for _ in range(burn_in):
            x_current, _ = _step_numba(
                x_current, R, h, mu1, sigma1, mu2, sigma2, weight
            )

        # Sampling
        for i in range(n_samples):
            x_current, accepted = _step_numba(
                x_current, R, h, mu1, sigma1, mu2, sigma2, weight
            )
            samples[i] = x_current
            if accepted:
                n_accepted += 1

        accept_rate = n_accepted / n_samples
        return samples, accept_rate


def compare_optimized_vs_original():
    """
    Compare optimized vs original implementation performance
    """
    np.random.seed(42)

    # Import original for comparison
    from SPS_bimodal import BimodalDistribution as OriginalBimodal
    from SPS_bimodal import StereographicProjectionSampler1D as OriginalSPS

    # Setup parameters
    n_samples = 50000
    x_init = 0.0

    # Original implementation
    print("=== ORIGINAL IMPLEMENTATION ===")
    original_sampler = OriginalSPS(R=1.5, h=0.6)
    original_bimodal = OriginalBimodal(
        mu1=-4, sigma1=0.8, mu2=3, sigma2=1.2, weight=0.4
    )

    start = timeit.default_timer()
    orig_samples, orig_accept = original_sampler.sample(
        x_init, original_bimodal.log_density, n_samples, burn_in=1000
    )
    orig_time = timeit.default_timer() - start

    print(f"Time: {orig_time:.3f}s ({n_samples / orig_time:.0f} samples/sec)")
    print(f"Acceptance rate: {orig_accept:.2%}")

    # Optimized implementation
    print("\n=== OPTIMIZED IMPLEMENTATION ===")
    opt_sampler = OptimizedStereographicProjectionSampler1D(R=1.5, h=0.6)
    opt_bimodal = OptimizedBimodalDistribution(
        mu1=-4, sigma1=0.8, mu2=3, sigma2=1.2, weight=0.4
    )

    # Reset seed for fair comparison
    np.random.seed(42)
    start = timeit.default_timer()
    opt_samples, opt_accept = opt_sampler.sample(
        x_init, opt_bimodal.log_density, n_samples, burn_in=1000
    )
    opt_time = timeit.default_timer() - start

    print(f"Time: {opt_time:.3f}s ({n_samples / opt_time:.0f} samples/sec)")
    print(f"Acceptance rate: {opt_accept:.2%}")
    print(f"Speedup: {orig_time / opt_time:.1f}x")

    if _HAVE_NUMBA:
        print("✓ Using Numba JIT compilation")
    else:
        print("✗ Numba not available, using Python fallback")

    # Statistical comparison
    print("\n=== STATISTICAL COMPARISON ===")
    print(
        f"Original mean: {np.mean(orig_samples):.3f}, std: {np.std(orig_samples):.3f}"
    )
    print(f"Optimized mean: {np.mean(opt_samples):.3f}, std: {np.std(opt_samples):.3f}")

    # Quick histogram comparison
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(orig_samples, bins=50, alpha=0.7, density=True, label="Original")
    plt.title("Original Implementation")
    plt.xlabel("x")
    plt.ylabel("Density")

    plt.subplot(1, 2, 2)
    plt.hist(
        opt_samples, bins=50, alpha=0.7, density=True, label="Optimized", color="orange"
    )
    plt.title("Optimized Implementation")
    plt.xlabel("x")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.show()

    return orig_samples, opt_samples, orig_time, opt_time


if __name__ == "__main__":
    compare_optimized_vs_original()
