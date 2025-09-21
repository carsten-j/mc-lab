import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import logsumexp

from mc_lab.pseudo_marginal_mcmc import PseudoMarginalMCMC

# EXAMPLE: Bayesian inference with intractable likelihood
# =========================================================
# Consider a model where:
# - Parameter θ ~ Normal(0, 1) [prior]
# - Latent variables U_i ~ Normal(θ, σ²) for i=1,...,M
# - Observed data Y_i ~ Normal(U_i, τ²)
#
# The marginal likelihood p(Y|θ) requires integrating out all U_i,
# which becomes intractable for large M.


def create_bayesian_example(y_obs, M=100, sigma2=1.0, tau2=0.5):
    """
    Creates an example where we need to marginalize over latent variables.

    The target is the posterior π(θ|Y) ∝ p(Y|θ) × p(θ)
    where p(Y|θ) = ∫ p(Y|U) p(U|θ) dU is intractable.

    We estimate p(Y|θ) using importance sampling in log-space.
    """

    def log_unbiased_likelihood_estimator(theta, n_samples=50):
        """
        Estimate log marginal likelihood log p(Y|θ) using importance sampling.
        This gives log of an unbiased estimate of the intractable integral.

        Note: While log(E[X]) ≠ E[log(X)], we compute E[X] first then take log.
        """
        theta = np.atleast_1d(theta)[0]

        # Use importance sampling with proposal q(U) = Normal(θ, σ² + τ²)
        var_proposal = sigma2 + tau2

        # Sample U from proposal
        u_samples = np.random.normal(theta, np.sqrt(var_proposal), (n_samples, M))

        # Compute log importance weights
        log_weights = np.zeros(n_samples)
        for i in range(n_samples):
            # log[p(Y|U) × p(U|θ) / q(U)]
            log_p_y_given_u = -0.5 * np.sum((y_obs - u_samples[i]) ** 2 / tau2)
            log_p_u_given_theta = -0.5 * np.sum((u_samples[i] - theta) ** 2 / sigma2)
            log_q_u = -0.5 * np.sum((u_samples[i] - theta) ** 2 / var_proposal)
            log_weights[i] = log_p_y_given_u + log_p_u_given_theta - log_q_u

        # Compute log of average weight using logsumexp for stability
        # log(mean(weights)) = log(sum(weights)/n) = logsumexp(log_weights) - log(n)
        log_estimate = logsumexp(log_weights) - np.log(n_samples)

        # Add log prior log p(θ) to get log unnormalized posterior
        log_prior = stats.norm.logpdf(theta, 0, 1)
        log_posterior_estimate = log_estimate + log_prior

        return log_posterior_estimate, None

    def proposal_sampler(theta_current):
        """Random walk proposal: θ* ~ Normal(θ_current, step_size²)"""
        step_size = 0.5
        return np.random.normal(theta_current, step_size, size=theta_current.shape)

    def log_proposal_density(theta_new, theta_old):
        """Evaluate log of random walk proposal density"""
        # step_size = 0.5
        # For symmetric proposal, this cancels in the ratio, so we return 0
        # But for completeness, here's the actual log density:
        # return stats.norm.logpdf(theta_new, theta_old, step_size)
        return 0.0  # Symmetric proposal

    return log_unbiased_likelihood_estimator, proposal_sampler, log_proposal_density


# Alternative example with asymmetric proposal for demonstration
def create_asymmetric_example(y_obs, M=100, sigma2=1.0, tau2=0.5):
    """
    Same model but with asymmetric proposal to show log_proposal_density in action.
    """

    def log_unbiased_likelihood_estimator(theta, n_samples=50):
        """Same as before"""
        theta = np.atleast_1d(theta)[0]
        var_proposal = sigma2 + tau2
        u_samples = np.random.normal(theta, np.sqrt(var_proposal), (n_samples, M))

        log_weights = np.zeros(n_samples)
        for i in range(n_samples):
            log_p_y_given_u = -0.5 * np.sum((y_obs - u_samples[i]) ** 2 / tau2)
            log_p_u_given_theta = -0.5 * np.sum((u_samples[i] - theta) ** 2 / sigma2)
            log_q_u = -0.5 * np.sum((u_samples[i] - theta) ** 2 / var_proposal)
            log_weights[i] = log_p_y_given_u + log_p_u_given_theta - log_q_u

        log_estimate = logsumexp(log_weights) - np.log(n_samples)
        log_prior = stats.norm.logpdf(theta, 0, 1)
        return log_estimate + log_prior, None

    def proposal_sampler(theta_current):
        """Asymmetric proposal: θ* ~ Normal(θ_current + drift, step_size²)"""
        step_size = 0.5
        drift = 0.1  # Slight drift upward
        return np.random.normal(
            theta_current + drift, step_size, size=theta_current.shape
        )

    def log_proposal_density(theta_new, theta_old):
        """Log density for asymmetric proposal"""
        step_size = 0.5
        drift = 0.1
        return stats.norm.logpdf(theta_new, theta_old + drift, step_size)

    return log_unbiased_likelihood_estimator, proposal_sampler, log_proposal_density


# Run example
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    M = 20  # Number of latent variables
    true_theta = 0.5
    sigma2 = 1.0
    tau2 = 0.5

    # Generate observed data
    u_true = np.random.normal(true_theta, np.sqrt(sigma2), M)
    y_obs = np.random.normal(u_true, np.sqrt(tau2))

    print(f"True θ: {true_theta}")
    print(f"Number of latent variables: {M}")
    print(f"Observed data shape: {y_obs.shape}")

    # Create the pseudo-marginal sampler (symmetric proposal)
    log_unbiased_est, prop_sampler, log_prop_density = create_bayesian_example(
        y_obs, M=M, sigma2=sigma2, tau2=tau2
    )

    pm_mcmc = PseudoMarginalMCMC(log_unbiased_est, prop_sampler, log_prop_density)

    # Run MCMC
    print("\nRunning Pseudo-Marginal MCMC with symmetric proposal...")
    samples, log_estimates, acc_rate = pm_mcmc.sample(
        x0=0.0,  # Start at prior mean
        n_samples=10000,
        verbose=True,
    )

    print(f"\nFinal acceptance rate: {acc_rate:.2%}")
    print(f"Posterior mean estimate: {np.mean(samples[2000:]):.3f}")
    print(f"Posterior std estimate: {np.std(samples[2000:]):.3f}")

    # Also run with asymmetric proposal for comparison
    print("\n" + "=" * 60)
    print("Testing asymmetric proposal...")
    log_unbiased_est_asym, prop_sampler_asym, log_prop_density_asym = (
        create_asymmetric_example(y_obs, M=M, sigma2=sigma2, tau2=tau2)
    )

    pm_mcmc_asym = PseudoMarginalMCMC(
        log_unbiased_est_asym, prop_sampler_asym, log_prop_density_asym
    )
    samples_asym, log_estimates_asym, acc_rate_asym = pm_mcmc_asym.sample(
        x0=0.0, n_samples=10000, verbose=False
    )

    print(f"Asymmetric proposal acceptance rate: {acc_rate_asym:.2%}")
    print(f"Asymmetric proposal posterior mean: {np.mean(samples_asym[2000:]):.3f}")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Trace plot - symmetric
    axes[0, 0].plot(samples[:, 0], alpha=0.7, label="Symmetric")
    axes[0, 0].axhline(true_theta, color="red", linestyle="--", label="True θ")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("θ")
    axes[0, 0].set_title("Trace Plot (Symmetric Proposal)")
    axes[0, 0].legend()

    # Histogram of samples - symmetric
    axes[0, 1].hist(
        samples[2000:, 0], bins=30, density=True, alpha=0.7, label="Symmetric"
    )
    axes[0, 1].axvline(true_theta, color="red", linestyle="--", label="True θ")
    axes[0, 1].set_xlabel("θ")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Posterior Distribution")
    axes[0, 1].legend()

    # Log estimates over time
    axes[0, 2].plot(log_estimates, alpha=0.7)
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("log(π̂)")
    axes[0, 2].set_title("Log Posterior Estimates")

    # Trace plot - asymmetric
    axes[1, 0].plot(samples_asym[:, 0], alpha=0.7, label="Asymmetric", color="orange")
    axes[1, 0].axhline(true_theta, color="red", linestyle="--", label="True θ")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("θ")
    axes[1, 0].set_title("Trace Plot (Asymmetric Proposal)")
    axes[1, 0].legend()

    # Compare histograms
    axes[1, 1].hist(
        samples[2000:, 0], bins=30, density=True, alpha=0.5, label="Symmetric"
    )
    axes[1, 1].hist(
        samples_asym[2000:, 0], bins=30, density=True, alpha=0.5, label="Asymmetric"
    )
    axes[1, 1].axvline(true_theta, color="red", linestyle="--", label="True θ")
    axes[1, 1].set_xlabel("θ")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Posterior Comparison")
    axes[1, 1].legend()

    # Autocorrelation comparison
    from statsmodels.tsa.stattools import acf

    autocorr_sym = acf(samples[2000:, 0], nlags=50)
    autocorr_asym = acf(samples_asym[2000:, 0], nlags=50)
    axes[1, 2].plot(autocorr_sym, label="Symmetric", alpha=0.7)
    axes[1, 2].plot(autocorr_asym, label="Asymmetric", alpha=0.7)
    axes[1, 2].set_xlabel("Lag")
    axes[1, 2].set_ylabel("Autocorrelation")
    axes[1, 2].set_title("Autocorrelation Function")
    axes[1, 2].axhline(0, color="black", linestyle="-", linewidth=0.5)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()

    # Print numerical stability demonstration
    print("\n" + "=" * 60)
    print("Numerical Stability Demonstration:")
    print("Log-space prevents underflow/overflow issues")
    print(f"Min log estimate: {np.min(log_estimates):.2f}")
    print(f"Max log estimate: {np.max(log_estimates):.2f}")
    print("If not in log space, these would be:")
    print(f"  Min: {np.exp(np.min(log_estimates)):.2e}")
    print(f"  Max: {np.exp(np.max(log_estimates)):.2e}")
    print("Working in log-space prevents numerical issues with these extreme values!")
