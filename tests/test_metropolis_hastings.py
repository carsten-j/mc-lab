"""
Comprehensive test suite for Metropolis-Hastings MCMC sampler.

Tests statistical correctness, ArviZ integration, convergence diagnostics,
and edge cases following MC-LAB testing patterns.
"""

import arviz as az
import numpy as np
import pytest
from scipy import stats

from mc_lab.metropolis_hastings import MetropolisHastingsSampler


class TestMetropolisHastingsSampler:
    """Test suite for Metropolis-Hastings sampler with comprehensive validation."""

    @pytest.fixture
    def random_seed(self):
        """Set random seed for reproducible tests."""
        np.random.seed(42)
        return 42

    def test_1d_standard_normal_sampling(self, random_seed):
        """
        Test Case 1: 1D Standard Normal Distribution

        Tests basic functionality with the simplest case - standard normal.
        Validates sample statistics, ArviZ integration, and acceptance rates.
        """

        def log_standard_normal(x):
            """Log PDF of standard normal distribution."""
            return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)

        sampler = MetropolisHastingsSampler(
            log_target=log_standard_normal,
            proposal_scale=0.8,
            adaptive_scaling=True,
        )

        # Generate samples
        n_samples = 2000
        idata = sampler.sample(
            n_samples=n_samples,
            n_chains=2,
            burn_in=500,
            thin=1,
            random_seed=random_seed,
            progressbar=False,
        )

        # Basic validation
        assert isinstance(idata, az.InferenceData), "Should return ArviZ InferenceData"
        assert "posterior" in idata.groups(), "Should have posterior group"
        assert "sample_stats" in idata.groups(), "Should have sample_stats group"

        # Check dimensions
        samples = idata.posterior["x"].values
        assert samples.shape == (2, n_samples), (
            f"Expected (2, {n_samples}), got {samples.shape}"
        )

        # Statistical validation
        all_samples = samples.flatten()
        sample_mean = np.mean(all_samples)
        sample_std = np.std(all_samples, ddof=1)

        # Mean should be close to 0
        assert abs(sample_mean) < 0.1, f"Sample mean {sample_mean} too far from 0"

        # Standard deviation should be close to 1
        assert abs(sample_std - 1.0) < 0.1, f"Sample std {sample_std} too far from 1"

        # Test sample_stats
        assert "log_likelihood" in idata.sample_stats, "Should track log_likelihood"
        assert "accepted" in idata.sample_stats, "Should track acceptance"
        assert "proposal_scale" in idata.sample_stats, "Should track proposal_scale"

        # Acceptance rate should be reasonable
        acceptance_rates = sampler.get_acceptance_rates(idata)
        assert 0.15 < acceptance_rates["overall"] < 0.7, (
            f"Acceptance rate {acceptance_rates['overall']:.3f} outside reasonable range"
        )

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.kstest(all_samples, stats.norm.cdf)
        assert p_value > 0.01, f"KS test failed: p-value {p_value:.4f} < 0.01"

        print(
            f"1D Normal - Mean: {sample_mean:.3f}, Std: {sample_std:.3f}, "
            f"Acceptance: {acceptance_rates['overall']:.3f}, KS p-value: {p_value:.3f}"
        )

    def test_2d_multivariate_normal_sampling(self, random_seed):
        """
        Test Case 2: 2D Multivariate Normal Distribution

        Tests multivariate sampling with correlation structure.
        Validates covariance estimation and variable naming.
        """
        # Target: MVN with correlation
        true_mean = np.array([1.0, -0.5])
        true_cov = np.array([[2.0, 0.8], [0.8, 1.5]])
        cov_inv = np.linalg.inv(true_cov)
        log_det_cov = np.log(np.linalg.det(true_cov))

        def log_mvn(x):
            """Log PDF of multivariate normal."""
            x = np.atleast_1d(x)
            diff = x - true_mean
            return -0.5 * (diff @ cov_inv @ diff + log_det_cov + 2 * np.log(2 * np.pi))

        sampler = MetropolisHastingsSampler(
            log_target=log_mvn,
            proposal_scale=np.array([0.7, 0.6]),  # Different scales per dimension
            var_names=["x1", "x2"],
            adaptive_scaling=True,
        )

        # Generate samples
        n_samples = 3000
        idata = sampler.sample(
            n_samples=n_samples,
            n_chains=3,
            burn_in=1000,
            thin=2,
            random_seed=random_seed,
            progressbar=False,
        )

        # Check variable names
        assert "x1" in idata.posterior, "Should have x1 variable"
        assert "x2" in idata.posterior, "Should have x2 variable"

        # Extract samples
        samples_x1 = idata.posterior["x1"].values
        samples_x2 = idata.posterior["x2"].values
        assert samples_x1.shape == (3, n_samples), f"x1 shape: {samples_x1.shape}"
        assert samples_x2.shape == (3, n_samples), f"x2 shape: {samples_x2.shape}"

        # Combine samples from all chains
        all_x1 = samples_x1.flatten()
        all_x2 = samples_x2.flatten()
        all_samples = np.column_stack([all_x1, all_x2])

        # Estimate sample statistics
        sample_mean = np.mean(all_samples, axis=0)
        sample_cov = np.cov(all_samples.T)

        # Validate means
        mean_error = np.abs(sample_mean - true_mean)
        assert np.all(mean_error < 0.15), f"Mean error {mean_error} too large"

        # Validate covariance
        cov_error = np.abs(sample_cov - true_cov)
        assert np.all(cov_error < 0.3), f"Covariance error too large:\n{cov_error}"

        # Check correlation
        sample_corr = sample_cov[0, 1] / np.sqrt(sample_cov[0, 0] * sample_cov[1, 1])
        true_corr = true_cov[0, 1] / np.sqrt(true_cov[0, 0] * true_cov[1, 1])
        assert abs(sample_corr - true_corr) < 0.1, (
            f"Correlation error: {abs(sample_corr - true_corr):.3f}"
        )

        # Test proposal scale adaptation (should be stored in sample_stats)
        proposal_scales = idata.sample_stats["proposal_scale"].values
        assert proposal_scales.shape == (3, n_samples, 2), (
            f"Proposal scale shape: {proposal_scales.shape}"
        )

        print(
            f"2D MVN - Mean error: {mean_error}, Corr error: {abs(sample_corr - true_corr):.3f}"
        )

    def test_challenging_bimodal_distribution(self, random_seed):
        """
        Test Case 3: Bimodal Gaussian Mixture

        Tests sampling from a challenging multimodal distribution.
        This tests the sampler's ability to explore multiple modes.
        """

        # Mixture of two 1D Gaussians: 0.6*N(-2, 0.8²) + 0.4*N(3, 1.2²)
        def log_bimodal(x):
            x = np.atleast_1d(x)[0]  # Handle scalar input

            # Component log-likelihoods
            log_comp1 = stats.norm.logpdf(x, loc=-2.0, scale=0.8)
            log_comp2 = stats.norm.logpdf(x, loc=3.0, scale=1.2)

            # Log of mixture: log(w1*p1 + w2*p2) = log(w1) + log_comp1 + log(1 + (w2/w1)*exp(log_comp2 - log_comp1))
            log_w1, log_w2 = np.log(0.6), np.log(0.4)

            # Numerically stable log-sum-exp
            if log_comp1 > log_comp2:
                return (
                    log_w1
                    + log_comp1
                    + np.log(1 + np.exp(log_w2 - log_w1 + log_comp2 - log_comp1))
                )
            else:
                return (
                    log_w2
                    + log_comp2
                    + np.log(1 + np.exp(log_w1 - log_w2 + log_comp1 - log_comp2))
                )

        sampler = MetropolisHastingsSampler(
            log_target=log_bimodal,
            proposal_scale=1.5,  # Larger scale to help mode jumping
            adaptive_scaling=True,
        )

        # Generate more samples for multimodal distribution
        n_samples = 5000
        idata = sampler.sample(
            n_samples=n_samples,
            n_chains=4,
            burn_in=2000,  # Longer burn-in
            thin=1,
            initial_states=np.array([-3, -1, 2, 4]).reshape(
                -1, 1
            ),  # Overdispersed starts
            random_seed=random_seed,
            progressbar=False,
        )

        # Extract samples
        samples = idata.posterior["x"].values.flatten()

        # Basic validation
        assert len(samples) == n_samples * 4, f"Expected {n_samples * 4} samples"

        # Check that both modes are sampled
        left_mode_samples = samples[samples < 0.5]  # Samples closer to left mode
        right_mode_samples = samples[samples > 0.5]  # Samples closer to right mode

        assert len(left_mode_samples) > 0.2 * len(samples), (
            "Too few samples from left mode"
        )
        assert len(right_mode_samples) > 0.2 * len(samples), (
            "Too few samples from right mode"
        )

        # Estimate mixture proportions
        prop_left = len(left_mode_samples) / len(samples)
        prop_right = len(right_mode_samples) / len(samples)

        # Should be roughly 60% left, 40% right (with some tolerance)
        assert abs(prop_left - 0.6) < 0.15, (
            f"Left mode proportion {prop_left:.3f} vs expected 0.6"
        )
        assert abs(prop_right - 0.4) < 0.15, (
            f"Right mode proportion {prop_right:.3f} vs expected 0.4"
        )

        # Check mode locations
        if len(left_mode_samples) > 100:
            left_mean = np.mean(left_mode_samples)
            assert abs(left_mean - (-2.0)) < 0.4, (
                f"Left mode mean {left_mean:.3f} vs expected -2.0"
            )

        if len(right_mode_samples) > 100:
            right_mean = np.mean(right_mode_samples)
            assert abs(right_mean - 3.0) < 0.4, (
                f"Right mode mean {right_mean:.3f} vs expected 3.0"
            )

        print(f"Bimodal - Left prop: {prop_left:.3f}, Right prop: {prop_right:.3f}")

    def test_adaptive_scaling_behavior(self, random_seed):
        """
        Test Case 4: Adaptive Scaling Functionality

        Tests that adaptive scaling properly adjusts proposal scales
        to achieve target acceptance rates.
        """

        def log_normal(x):
            return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)

        # Test with very small initial scale (should increase)
        sampler_small = MetropolisHastingsSampler(
            log_target=log_normal,
            proposal_scale=0.01,  # Very small
            adaptive_scaling=True,
            target_acceptance_rate=0.4,
        )

        idata_small = sampler_small.sample(
            n_samples=1000,
            n_chains=1,
            burn_in=500,
            random_seed=random_seed,
            progressbar=False,
        )

        # Test with very large initial scale (should decrease)
        sampler_large = MetropolisHastingsSampler(
            log_target=log_normal,
            proposal_scale=10.0,  # Very large
            adaptive_scaling=True,
            target_acceptance_rate=0.4,
        )

        idata_large = sampler_large.sample(
            n_samples=1000,
            n_chains=1,
            burn_in=500,
            random_seed=random_seed + 1,
            progressbar=False,
        )

        # Check that scales adapted
        initial_scale_small = idata_small.sample_stats["proposal_scale"].values[0, 0, 0]
        final_scale_small = idata_small.sample_stats["proposal_scale"].values[0, -1, 0]

        initial_scale_large = idata_large.sample_stats["proposal_scale"].values[0, 0, 0]
        final_scale_large = idata_large.sample_stats["proposal_scale"].values[0, -1, 0]

        # Small initial scale should increase
        assert final_scale_small > initial_scale_small * 1.5, (
            f"Small scale didn't increase: {initial_scale_small:.4f} -> {final_scale_small:.4f}"
        )

        # Large initial scale should decrease
        assert final_scale_large < initial_scale_large * 0.5, (
            f"Large scale didn't decrease: {initial_scale_large:.4f} -> {final_scale_large:.4f}"
        )

        # Acceptance rates should be closer to target
        accept_rate_small = sampler_small.get_acceptance_rates(idata_small)["overall"]
        accept_rate_large = sampler_large.get_acceptance_rates(idata_large)["overall"]

        assert 0.25 < accept_rate_small < 0.55, (
            f"Small scale acceptance rate: {accept_rate_small:.3f}"
        )
        assert 0.25 < accept_rate_large < 0.55, (
            f"Large scale acceptance rate: {accept_rate_large:.3f}"
        )

        print(
            f"Adaptive - Small: {initial_scale_small:.4f} -> {final_scale_small:.4f} "
            f"(accept: {accept_rate_small:.3f})"
        )
        print(
            f"Adaptive - Large: {initial_scale_large:.4f} -> {final_scale_large:.4f} "
            f"(accept: {accept_rate_large:.3f})"
        )

    def test_convergence_diagnostics_with_arviz(self, random_seed):
        """
        Test Case 5: ArviZ Integration and Convergence Diagnostics

        Tests that the sampler produces data compatible with ArviZ diagnostics
        and achieves good convergence for simple problems.
        """

        def log_normal_2d(x):
            """2D independent normal for easy convergence."""
            return -0.5 * np.sum(x**2) - np.log(2 * np.pi)

        sampler = MetropolisHastingsSampler(
            log_target=log_normal_2d,
            proposal_scale=np.array([0.8, 0.8]),
            var_names=["theta1", "theta2"],
        )

        # Generate samples with multiple chains for diagnostics
        idata = sampler.sample(
            n_samples=1000,
            n_chains=4,
            burn_in=500,
            thin=1,
            random_seed=random_seed,
            progressbar=False,
        )

        # Test ArviZ functionality
        summary = az.summary(idata)
        assert summary is not None, "ArviZ summary should work"
        assert len(summary) == 2, "Should have 2 variables in summary"

        # Test R-hat (should be close to 1 for converged chains)
        rhat = az.rhat(idata)
        for var in ["theta1", "theta2"]:
            assert var in rhat, f"R-hat should include {var}"
            assert rhat[var].values < 1.1, (
                f"R-hat for {var}: {rhat[var].values:.4f} > 1.1"
            )

        # Test effective sample size
        ess = az.ess(idata)
        for var in ["theta1", "theta2"]:
            assert var in ess, f"ESS should include {var}"
            assert ess[var].values > 100, (
                f"ESS for {var}: {ess[var].values:.0f} too low"
            )

        # Test MCSE (Monte Carlo Standard Error)
        mcse = az.mcse(idata)
        for var in ["theta1", "theta2"]:
            assert var in mcse, f"MCSE should include {var}"
            assert mcse[var].values < 0.1, (
                f"MCSE for {var}: {mcse[var].values:.4f} too high"
            )

        print(
            f"Convergence - R-hat: theta1={rhat['theta1'].values:.4f}, "
            f"theta2={rhat['theta2'].values:.4f}"
        )
        print(
            f"Convergence - ESS: theta1={ess['theta1'].values:.0f}, "
            f"theta2={ess['theta2'].values:.0f}"
        )

    def test_edge_cases_and_error_handling(self, random_seed):
        """
        Test Case 6: Edge Cases and Error Handling

        Tests sampler robustness with edge cases, invalid inputs,
        and numerical issues.
        """

        def log_normal(x):
            return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)

        # Test invalid proposal scale dimensions
        with pytest.raises(ValueError, match="doesn't match dimension"):
            sampler = MetropolisHastingsSampler(
                log_target=log_normal,
                proposal_scale=np.array([1.0, 2.0]),  # 2D scale for 1D problem
                var_names=["x"],
            )
            sampler.sample(100, progressbar=False)

        # Test invalid variable names
        with pytest.raises(ValueError, match="doesn't match dimension"):
            sampler = MetropolisHastingsSampler(
                log_target=log_normal,
                var_names=["x", "y"],  # 2 names for 1D problem
            )
            sampler.sample(100, progressbar=False)

        # Test with target function that raises exceptions
        def bad_log_target(x):
            if x > 5:
                raise ValueError("Bad region")
            return log_normal(x)

        sampler = MetropolisHastingsSampler(
            log_target=bad_log_target,
            proposal_scale=2.0,  # Large scale to hit bad region
        )

        # Should handle exceptions gracefully (reject bad proposals)
        idata = sampler.sample(
            n_samples=500,
            n_chains=1,
            burn_in=200,
            initial_states=np.array([[0.0]]),  # Start in good region
            random_seed=random_seed,
            progressbar=False,
        )

        # Should still get reasonable samples
        samples = idata.posterior["x"].values.flatten()
        assert len(samples) == 500, "Should get requested number of samples"
        assert np.all(samples < 10), "Should avoid bad regions"

        # Test zero sample request
        idata_empty = sampler.sample(
            n_samples=0,
            n_chains=1,
            burn_in=0,
            progressbar=False,
        )

        assert idata_empty.posterior["x"].shape == (1, 0), "Should handle zero samples"

    @pytest.mark.performance
    def test_performance_and_scaling(self, random_seed):
        """
        Performance test for larger sample sizes and higher dimensions.

        This test is marked as performance and checks that the sampler
        can handle realistic problem sizes efficiently.
        """

        def log_mvn_5d(x):
            """5D independent normal distribution."""
            return -0.5 * np.sum(x**2) - 2.5 * np.log(2 * np.pi)

        sampler = MetropolisHastingsSampler(
            log_target=log_mvn_5d,
            proposal_scale=0.8,
            adaptive_scaling=True,
        )

        # Larger sample size test
        import time

        start_time = time.time()

        idata = sampler.sample(
            n_samples=5000,
            n_chains=2,
            burn_in=1000,
            thin=1,
            random_seed=random_seed,
            progressbar=False,
        )

        elapsed_time = time.time() - start_time

        # Basic validation
        assert idata.posterior.x0.shape == (2, 5000), "Correct output shape"

        # Should complete in reasonable time (< 30 seconds on most machines)
        assert elapsed_time < 30, f"Sampling took {elapsed_time:.1f} seconds, too slow"

        # Sample quality check
        samples = np.column_stack(
            [idata.posterior[f"x{i}"].values.flatten() for i in range(5)]
        )

        sample_means = np.mean(samples, axis=0)
        sample_stds = np.std(samples, axis=0, ddof=1)

        # All dimensions should be approximately N(0,1)
        assert np.all(np.abs(sample_means) < 0.1), (
            f"Means too far from 0: {sample_means}"
        )
        assert np.all(np.abs(sample_stds - 1.0) < 0.15), (
            f"Stds too far from 1: {sample_stds}"
        )

        print(f"Performance - 5D, 10K samples, 2 chains: {elapsed_time:.2f} seconds")
        print(f"Performance - Final means: {sample_means}")

    def test_tune_proposal_scale_method(self, random_seed):
        """
        Test the standalone proposal scale tuning method.
        """

        def log_normal(x):
            return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)

        sampler = MetropolisHastingsSampler(
            log_target=log_normal,
            proposal_scale=1.0,  # Will be overridden by tuning
        )

        # Tune from a bad initial scale
        tuned_scale = sampler.tune_proposal_scale(
            initial_scale=0.01,  # Very small
            target_samples=500,
            random_seed=random_seed,
        )

        assert isinstance(tuned_scale, float), "Should return float for 1D"
        assert tuned_scale > 0.01, f"Should increase small scale: {tuned_scale}"
        assert tuned_scale < 5.0, f"Should not be too large: {tuned_scale}"

        # Test the tuned scale gives reasonable acceptance rate
        sampler.proposal_scale = np.array([tuned_scale])
        sampler.adaptive_scaling = False  # Use fixed tuned scale

        idata = sampler.sample(
            n_samples=1000,
            n_chains=1,
            burn_in=100,
            random_seed=random_seed + 1,
            progressbar=False,
        )

        acceptance_rate = sampler.get_acceptance_rates(idata)["overall"]
        assert 0.2 < acceptance_rate < 0.6, (
            f"Tuned scale gives poor acceptance rate: {acceptance_rate:.3f}"
        )

        print(
            f"Tuning - Tuned scale: {tuned_scale:.3f}, "
            f"Acceptance rate: {acceptance_rate:.3f}"
        )
