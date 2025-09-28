"""
Tests for Barker's MCMC sampler.

This module tests the basic functionality of the BarkerMCMCSampler,
including sampling from simple distributions and comparing with
expected statistical properties.
"""

import numpy as np
import pytest

from mc_lab.barker_mcmc import BarkerMCMCSampler


class TestBarkerMCMCSampler:
    """Test cases for BarkerMCMCSampler."""

    def test_1d_standard_normal_sampling(self):
        """Test sampling from 1D standard normal distribution."""

        def log_standard_normal(x):
            """Log density of standard normal distribution."""
            x = np.asarray(x).flatten()
            if x.size == 1:
                val = x[0]
                return float(-0.5 * val**2 - 0.5 * np.log(2 * np.pi))
            else:
                raise ValueError("Expected 1D input")

        sampler = BarkerMCMCSampler(
            log_target=log_standard_normal,
            proposal_scale=0.8,
            adaptive_scaling=True,
            target_acceptance_rate=0.5,
        )

        # Sample from the distribution
        idata = sampler.sample(
            n_samples=500,
            n_chains=2,
            burn_in=200,
            random_seed=42,
            progressbar=False,
        )

        # Check that we have the right structure
        assert "x" in idata.posterior
        assert idata.posterior["x"].shape == (2, 500)  # n_chains, n_samples

        # Check sample statistics are reasonable (within 3 sigma of expected)
        samples = idata.posterior["x"].values.flatten()
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)

        # For standard normal: mean ≈ 0, std ≈ 1
        assert abs(sample_mean) < 0.2, f"Sample mean {sample_mean} too far from 0"
        assert abs(sample_std - 1.0) < 0.2, f"Sample std {sample_std} too far from 1"

    def test_2d_mvn_sampling(self):
        """Test sampling from 2D multivariate normal."""

        # Target: bivariate normal with mean [1, -0.5] and some correlation
        mu = np.array([1.0, -0.5])
        cov = np.array([[1.5, 0.3], [0.3, 0.8]])
        cov_inv = np.linalg.inv(cov)

        def log_mvn(x):
            """Log density of 2D multivariate normal."""
            x = np.asarray(x)
            if x.size != 2:
                raise ValueError("Expected 2D input")
            diff = x - mu
            return float(-0.5 * diff @ cov_inv @ diff)

        sampler = BarkerMCMCSampler(
            log_target=log_mvn,
            proposal_scale=[0.8, 0.6],
            var_names=["x", "y"],
            adaptive_scaling=True,
        )

        # Sample from the distribution
        idata = sampler.sample(
            n_samples=600,
            n_chains=2,
            burn_in=300,
            random_seed=123,
            progressbar=False,
        )

        # Check structure
        assert "x" in idata.posterior
        assert "y" in idata.posterior
        assert idata.posterior["x"].shape == (2, 600)
        assert idata.posterior["y"].shape == (2, 600)

        # Check sample means are reasonable
        x_samples = idata.posterior["x"].values.flatten()
        y_samples = idata.posterior["y"].values.flatten()

        x_mean = np.mean(x_samples)
        y_mean = np.mean(y_samples)

        assert abs(x_mean - mu[0]) < 0.3, f"X mean {x_mean} too far from {mu[0]}"
        assert abs(y_mean - mu[1]) < 0.3, f"Y mean {y_mean} too far from {mu[1]}"

    def test_acceptance_rates_tracking(self):
        """Test that acceptance rates are tracked correctly."""

        def log_standard_normal(x):
            x = np.asarray(x).flatten()
            if x.size == 1:
                val = x[0]
                return float(-0.5 * val**2 - 0.5 * np.log(2 * np.pi))
            else:
                raise ValueError("Expected 1D input")

        sampler = BarkerMCMCSampler(
            log_target=log_standard_normal,
            proposal_scale=1.0,
            adaptive_scaling=False,  # Fixed scale to test acceptance rate
        )

        idata = sampler.sample(
            n_samples=200,
            n_chains=1,
            burn_in=100,
            random_seed=456,
            progressbar=False,
        )

        # Check acceptance rate tracking
        acceptance_rates = sampler.get_acceptance_rates(idata)

        assert "chain_0" in acceptance_rates
        assert "overall" in acceptance_rates
        assert 0.0 <= acceptance_rates["overall"] <= 1.0

        # For Barker MCMC with reasonable proposal scale, acceptance should be > 0
        assert acceptance_rates["overall"] > 0.1

    def test_barker_specific_acceptance_probs(self):
        """Test that Barker-specific acceptance probabilities are tracked."""

        def log_standard_normal(x):
            x = np.asarray(x).flatten()
            if x.size == 1:
                val = x[0]
                return float(-0.5 * val**2 - 0.5 * np.log(2 * np.pi))
            else:
                raise ValueError("Expected 1D input")

        sampler = BarkerMCMCSampler(
            log_target=log_standard_normal,
            proposal_scale=0.5,
            adaptive_scaling=False,
        )

        idata = sampler.sample(
            n_samples=100,
            n_chains=1,
            burn_in=50,
            random_seed=789,
            progressbar=False,
        )

        # Check that Barker acceptance probabilities are stored
        assert "acceptance_prob" in idata.sample_stats

        barker_probs = sampler.get_barker_acceptance_probs(idata)
        assert "chain_0" in barker_probs
        assert "overall" in barker_probs
        assert 0.0 <= barker_probs["overall"] <= 1.0

    def test_custom_variable_names(self):
        """Test that custom variable names work correctly."""

        def log_2d_target(x):
            # Ensure x is treated as 2D
            x = np.asarray(x)
            if x.size != 2:
                raise ValueError("Expected 2D input")
            return float(-0.5 * np.sum(x**2))

        custom_names = ["alpha", "beta"]
        sampler = BarkerMCMCSampler(
            log_target=log_2d_target,
            proposal_scale=[1.0, 1.0],  # Specify 2D proposal
            var_names=custom_names,
        )

        idata = sampler.sample(
            n_samples=50,
            n_chains=1,
            burn_in=25,
            random_seed=999,
            progressbar=False,
        )

        # Check that custom names are used
        assert "alpha" in idata.posterior
        assert "beta" in idata.posterior
        assert "x0" not in idata.posterior
        assert "x1" not in idata.posterior

    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""

        def log_target_simple(x):
            x = np.asarray(x)
            return float(-0.5 * np.sum(x**2))

        # Test mismatched proposal_scale dimension
        # Set up a 1D target explicitly but provide multi-dimensional proposal_scale
        with pytest.raises(ValueError, match="proposal_scale length"):
            sampler = BarkerMCMCSampler(
                log_target=log_target_simple,
                proposal_scale=[1.0, 2.0, 3.0],  # 3 scales
                var_names=["x"],  # But only 1 variable name (forces 1D)
            )
            sampler.sample(
                n_samples=10,
                n_chains=1,
                burn_in=5,
                random_seed=111,
                progressbar=False,
            )

        # Test mismatched var_names dimension
        with pytest.raises(ValueError, match="var_names length"):
            sampler = BarkerMCMCSampler(
                log_target=log_target_simple,
                proposal_scale=1.0,  # Single scale
                var_names=[
                    "x",
                    "y",
                    "z",
                ],  # But 3 variable names - this should fail during dimension check
            )
            # Need to provide initial states to avoid dimension detection from target
            initial_states = np.array([[1.0]])  # 1D initial state
            sampler.sample(
                n_samples=10,
                n_chains=1,
                burn_in=5,
                initial_states=initial_states,
                random_seed=222,
                progressbar=False,
            )
