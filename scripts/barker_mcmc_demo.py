"""
Demo script for Barker's MCMC sampler.

This script demonstrates how to use the BarkerMCMCSampler to sample from
different probability distributions and compares it with standard
Metropolis-Hastings sampling.
"""

import arviz as az
import numpy as np

from mc_lab import BarkerMCMCSampler, MetropolisHastingsSampler


def main():
    """Run demonstrations of Barker MCMC sampling."""

    print("=== Barker MCMC Sampler Demo ===\n")

    # Example 1: 1D Normal Distribution
    print("1. Sampling from 1D Standard Normal Distribution")
    print("-" * 50)

    def log_standard_normal(x):
        """Log density of standard normal distribution."""
        x = np.asarray(x).flatten()
        if x.size == 1:
            val = x[0]
            return float(-0.5 * val**2 - 0.5 * np.log(2 * np.pi))
        else:
            raise ValueError("Expected 1D input")

    # Compare Barker vs Metropolis-Hastings
    barker_sampler = BarkerMCMCSampler(
        log_target=log_standard_normal,
        proposal_scale=0.8,
        target_acceptance_rate=0.5,  # Barker typically has higher optimal acceptance
    )

    mh_sampler = MetropolisHastingsSampler(
        log_target=log_standard_normal,
        proposal_scale=0.8,
        target_acceptance_rate=0.35,  # MH optimal acceptance
    )

    print("Sampling with Barker MCMC...")
    barker_idata = barker_sampler.sample(
        n_samples=1000,
        n_chains=2,
        burn_in=500,
        random_seed=42,
        progressbar=False,
    )

    print("\nSampling with Metropolis-Hastings...")
    mh_idata = mh_sampler.sample(
        n_samples=1000,
        n_chains=2,
        burn_in=500,
        random_seed=42,
        progressbar=False,
    )

    # Print summaries
    print("\nBarker MCMC Results:")
    print(az.summary(barker_idata, round_to=4))

    print("\nMetropolis-Hastings Results:")
    print(az.summary(mh_idata, round_to=4))

    # Compare acceptance rates
    barker_rates = barker_sampler.get_acceptance_rates(barker_idata)
    mh_rates = mh_sampler.get_acceptance_rates(mh_idata)

    print(f"\nBarker acceptance rate: {barker_rates['overall']:.3f}")
    print(f"Metropolis-Hastings acceptance rate: {mh_rates['overall']:.3f}")

    # Example 2: 2D Correlated Normal
    print("\n\n2. Sampling from 2D Correlated Normal Distribution")
    print("-" * 60)

    # Target: bivariate normal with correlation
    mu = np.array([1.0, -0.5])
    cov = np.array([[1.5, 0.6], [0.6, 1.0]])
    cov_inv = np.linalg.inv(cov)

    def log_mvn(x):
        """Log density of 2D multivariate normal."""
        x = np.asarray(x)
        if x.size != 2:
            raise ValueError("Expected 2D input")
        diff = x - mu
        return float(-0.5 * diff @ cov_inv @ diff)

    barker_mvn_sampler = BarkerMCMCSampler(
        log_target=log_mvn,
        proposal_scale=[0.7, 0.8],
        var_names=["x", "y"],
        adaptive_scaling=True,
    )

    print("Sampling from 2D correlated normal with Barker MCMC...")
    mvn_idata = barker_mvn_sampler.sample(
        n_samples=1500,
        n_chains=2,
        burn_in=750,
        random_seed=123,
        progressbar=False,
    )

    print("\n2D Barker MCMC Results:")
    print(az.summary(mvn_idata, round_to=4))

    # Show some Barker-specific diagnostics
    barker_probs = barker_mvn_sampler.get_barker_acceptance_probs(mvn_idata)
    print(f"\nBarker acceptance probabilities: {barker_probs['overall']:.3f}")

    # Example 3: Demonstrate adaptive scaling
    print("\n\n3. Adaptive Proposal Scaling Demo")
    print("-" * 40)

    # Start with a poor proposal scale
    adaptive_sampler = BarkerMCMCSampler(
        log_target=log_standard_normal,
        proposal_scale=5.0,  # Too large initially
        adaptive_scaling=True,
        target_acceptance_rate=0.5,
    )

    print("Sampling with adaptive scaling (starting with poor proposal scale)...")
    adaptive_idata = adaptive_sampler.sample(
        n_samples=800,
        n_chains=1,
        burn_in=1000,  # Longer burn-in for adaptation
        random_seed=456,
        progressbar=False,
    )

    print("\nAdaptive Scaling Results:")
    print(az.summary(adaptive_idata, round_to=4))

    adaptive_rates = adaptive_sampler.get_acceptance_rates(adaptive_idata)
    print(f"Final acceptance rate: {adaptive_rates['overall']:.3f}")

    print("\n=== Demo Complete ===")
    print("\nKey advantages of Barker MCMC:")
    print("• Often better mixing than standard Metropolis-Hastings")
    print("• Higher optimal acceptance rates (≈0.5 vs ≈0.35)")
    print("• Can be more efficient for multi-modal distributions")
    print(
        "• Uses acceptance probability π(x')/(π(x') + π(x)) instead of min(1, π(x')/π(x))"
    )


if __name__ == "__main__":
    main()
