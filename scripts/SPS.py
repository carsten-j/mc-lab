import matplotlib.pyplot as plt
import numpy as np


class StereographicProjectionSampler:
    """
    Stereographic Projection Sampler (SPS) for MCMC sampling
    """

    def __init__(self, R=1.0, h=0.5):
        """
        Parameters:
        R: radius parameter for stereographic projection
        h: step size for random walk on sphere
        """
        self.R = R
        self.h = h

    def SP_inverse(self, x):
        """
        Inverse stereographic projection: R^d -> S^d \ {North Pole}
        Maps from Euclidean space to unit sphere

        From equation (3) in the paper:
        z_i = 2Rx_i / (||x||^2 + R^2) for i = 1,...,d
        z_{d+1} = (||x||^2 - R^2) / (||x||^2 + R^2)
        """
        x = np.atleast_1d(x)
        d = len(x)
        z = np.zeros(d + 1)

        norm_x_sq = np.sum(x**2)
        denom = norm_x_sq + self.R**2

        # First d components
        z[:d] = 2 * self.R * x / denom
        # Last component
        z[d] = (norm_x_sq - self.R**2) / denom

        return z

    def SP(self, z):
        """
        Stereographic projection: S^d \ {North Pole} -> R^d
        Maps from unit sphere to Euclidean space

        x = SP(z) = (R*z_1/(1-z_{d+1}), ..., R*z_d/(1-z_{d+1}))^T
        """
        d = len(z) - 1
        x = np.zeros(d)

        denom = 1 - z[-1]  # 1 - z_{d+1}
        if abs(denom) < 1e-10:
            # Handle near north pole
            return np.ones(d) * np.inf

        x = self.R * z[:d] / denom

        return x

    def propose_on_sphere(self, z):
        """
        Propose new point on sphere using tangent space random walk
        """
        d_plus_1 = len(z)

        # Sample random perturbation in R^{d+1}
        d_tilde_z = np.random.normal(0, self.h, d_plus_1)

        # Project to tangent space at z
        z_norm_sq = np.sum(z**2)
        dz = d_tilde_z - (np.dot(z, d_tilde_z) / z_norm_sq) * z

        # Re-project to sphere
        z_new = z + dz
        z_new = z_new / np.linalg.norm(z_new)

        return z_new

    def step(self, x_current, log_posterior_func):
        """
        One step of the SPS algorithm

        Parameters:
        x_current: current state in R^d
        log_posterior_func: function that computes log posterior density

        Returns:
        x_next: next state
        accepted: whether proposal was accepted
        """
        d = len(x_current)

        # Step 1: Map current state to sphere
        z = self.SP_inverse(x_current)

        # Step 2: Propose new point on sphere
        z_proposed = self.propose_on_sphere(z)

        # Step 3: Map back to R^d
        x_proposed = self.SP(z_proposed)

        # Check for invalid proposals (near north pole)
        if np.any(np.isinf(x_proposed)) or np.any(np.isnan(x_proposed)):
            return x_current, False

        # Step 4: Compute acceptance probability
        # α = min(1, [π(X̂)(R^2 + ||X̂||^2)^d] / [π(x)(R^2 + ||x||^2)^d])

        norm_x_current_sq = np.sum(x_current**2)
        norm_x_proposed_sq = np.sum(x_proposed**2)

        log_ratio = (
            log_posterior_func(x_proposed)
            - log_posterior_func(x_current)
            + d
            * (
                np.log(self.R**2 + norm_x_proposed_sq)
                - np.log(self.R**2 + norm_x_current_sq)
            )
        )

        accept_prob = min(1.0, np.exp(log_ratio))

        # Step 5: Accept or reject
        if np.random.rand() < accept_prob:
            return x_proposed, True
        else:
            return x_current, False

    def sample(self, x_init, log_posterior_func, n_samples, burn_in=1000):
        """
        Generate MCMC samples using SPS

        Parameters:
        x_init: initial state
        log_posterior_func: function that computes log posterior density
        n_samples: number of samples to generate
        burn_in: number of burn-in samples

        Returns:
        samples: array of samples
        accept_rate: acceptance rate
        """
        d = len(x_init)
        samples = np.zeros((n_samples, d))
        x_current = x_init.copy()
        n_accepted = 0

        # Burn-in
        for _ in range(burn_in):
            x_current, accepted = self.step(x_current, log_posterior_func)

        # Sampling
        for i in range(n_samples):
            x_current, accepted = self.step(x_current, log_posterior_func)
            samples[i] = x_current
            if accepted:
                n_accepted += 1

        accept_rate = n_accepted / n_samples

        return samples, accept_rate


class CauchyRegression:
    """
    Cauchy regression model with posterior from equation (1)
    Y_i ~ Cauchy(α + β^T X_i, γ)
    """

    def __init__(self, X, Y, a=1.0, b=1.0):
        """
        Parameters:
        X: design matrix (n x p)
        Y: response vector (n,)
        a, b: parameters for Gamma prior on γ
        """
        self.X = X
        self.Y = Y
        self.n = len(Y)
        self.p = X.shape[1] if len(X.shape) > 1 else 1
        self.a = a
        self.b = b

    def log_posterior(self, params):
        """
        Compute log posterior density from equation (1)

        Parameters:
        params: [α, β_1, ..., β_p, log(γ)]

        Note: We parameterize with log(γ) for unconstrained optimization
        """
        if len(params) != self.p + 2:
            raise ValueError("Invalid parameter dimension")

        alpha = params[0]
        beta = params[1 : self.p + 1] if self.p > 0 else np.array([])
        log_gamma = params[-1]
        gamma = np.exp(log_gamma)

        # Prior for gamma: Gamma(a, b)
        log_prior = (self.a - 1) * log_gamma - self.b * gamma

        # Likelihood: Cauchy
        if self.p > 0:
            mu = alpha + np.dot(self.X, beta)
        else:
            mu = alpha * np.ones(self.n)

        residuals = (self.Y - mu) / gamma
        log_likelihood = -np.sum(np.log(1 + residuals**2))

        # Add Jacobian for log transformation
        log_posterior = log_prior + log_likelihood + log_gamma

        return log_posterior


# Example usage with synthetic data
def demo_sps_cauchy():
    """
    Demonstrate SPS on Cauchy regression model
    """
    np.random.seed(42)

    # Generate synthetic data
    n = 50
    p = 2  # number of covariates

    # True parameters
    true_alpha = 2.0
    true_beta = np.array([1.5, -0.5])
    true_gamma = 1.0

    # Design matrix
    X = np.random.randn(n, p)

    # Generate Cauchy responses
    mu_true = true_alpha + np.dot(X, true_beta)
    Y = mu_true + true_gamma * np.random.standard_cauchy(n)

    # Set up model
    model = CauchyRegression(X, Y, a=1.0, b=0.1)

    # Set up sampler
    d = p + 2  # dimension: alpha + beta + log(gamma)
    R = np.sqrt(d)  # Recommended radius from paper
    sampler = StereographicProjectionSampler(R=R, h=0.3)

    # Initial values
    x_init = np.zeros(d)
    x_init[-1] = 0.0  # log(gamma) = 0 => gamma = 1

    # Run sampler
    print("Running Stereographic Projection Sampler...")
    samples, accept_rate = sampler.sample(
        x_init, model.log_posterior, n_samples=5000, burn_in=1000
    )

    print(f"Acceptance rate: {accept_rate:.2%}")

    # Extract parameters
    alpha_samples = samples[:, 0]
    beta_samples = samples[:, 1 : p + 1]
    gamma_samples = np.exp(samples[:, -1])

    # Print results
    print("\nPosterior means:")
    print(f"α: {np.mean(alpha_samples):.3f} (true: {true_alpha})")
    for i in range(p):
        print(f"β_{i + 1}: {np.mean(beta_samples[:, i]):.3f} (true: {true_beta[i]})")
    print(f"γ: {np.mean(gamma_samples):.3f} (true: {true_gamma})")

    # Plot traces
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(alpha_samples)
    axes[0, 0].set_title("α (intercept)")
    axes[0, 0].axhline(true_alpha, color="red", linestyle="--", label="True value")
    axes[0, 0].legend()

    axes[0, 1].plot(beta_samples[:, 0])
    axes[0, 1].set_title("β₁")
    axes[0, 1].axhline(true_beta[0], color="red", linestyle="--", label="True value")
    axes[0, 1].legend()

    axes[1, 0].plot(beta_samples[:, 1] if p > 1 else np.zeros_like(alpha_samples))
    axes[1, 0].set_title("β₂")
    if p > 1:
        axes[1, 0].axhline(
            true_beta[1], color="red", linestyle="--", label="True value"
        )
    axes[1, 0].legend()

    axes[1, 1].plot(gamma_samples)
    axes[1, 1].set_title("γ (scale)")
    axes[1, 1].axhline(true_gamma, color="red", linestyle="--", label="True value")
    axes[1, 1].legend()

    plt.suptitle("SPS Trace Plots for Cauchy Regression")
    plt.tight_layout()
    plt.show()

    return samples, accept_rate


if __name__ == "__main__":
    samples, accept_rate = demo_sps_cauchy()
