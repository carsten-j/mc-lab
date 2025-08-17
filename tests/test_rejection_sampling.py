import numpy as np
import pytest
from scipy import stats
from scipy.stats import kstest

from mc_lab.rejection_sampling import RejectionSampler


class TestRejectionSampling:
    """Comprehensive test suite for rejection sampling with non-trivial cases."""

    @pytest.fixture
    def random_seed(self):
        """Set random seed for reproducible tests."""
        np.random.seed(42)

    def test_bimodal_mixture_distribution(self, random_seed):
        """
        Test Case 1: Bimodal Gaussian Mixture

        Tests sampling from a complex bimodal distribution (mixture of two Gaussians)
        using a single Gaussian proposal. This is challenging because:
        - The target has two modes at different locations
        - Single Gaussian proposal is a poor fit
        - Tests adaptive M estimation and low acceptance rates
        - Validates that both modes are properly sampled
        """

        # Target: Mixture of N(-2, 0.5²) and N(3, 0.8²) with weights 0.4, 0.6
        def bimodal_pdf(x):
            component1 = 0.4 * stats.norm.pdf(x, loc=-2, scale=0.5)
            component2 = 0.6 * stats.norm.pdf(x, loc=3, scale=0.8)
            return component1 + component2

        # Proposal: N(0, 2²) - broad normal centered between modes
        def proposal_pdf(x):
            return stats.norm.pdf(x, loc=0, scale=2)

        def proposal_sampler(n):
            return np.random.normal(0, 2, n)

        # Create sampler with adaptive M estimation
        sampler = RejectionSampler(
            target_pdf=bimodal_pdf,
            proposal_pdf=proposal_pdf,
            proposal_sampler=proposal_sampler,
            adaptive_M=True,
            initial_batch_size=2000,
        )

        # Generate large sample to capture both modes
        n_samples = 10000
        samples = sampler.sample(n_samples)
        stats_dict = sampler.get_stats()

        # Basic validation
        assert len(samples) == n_samples, "Incorrect number of samples generated"
        assert stats_dict["acceptance_rate"] > 0.01, "Acceptance rate too low"
        assert stats_dict["current_M"] > 1.0, "M should be greater than 1"

        # Test bimodality - samples should cluster around -2 and 3
        samples_left = samples[samples < 0.5]  # Closer to left mode
        samples_right = samples[samples > 0.5]  # Closer to right mode

        assert len(samples_left) > 0.2 * n_samples, "Left mode not properly sampled"
        assert len(samples_right) > 0.4 * n_samples, "Right mode not properly sampled"

        # Check mode locations (within tolerance)
        left_mode_mean = np.mean(samples_left)
        right_mode_mean = np.mean(samples_right)

        assert abs(left_mode_mean - (-2)) < 0.3, (
            f"Left mode mean {left_mode_mean} too far from -2"
        )
        assert abs(right_mode_mean - 3) < 0.3, (
            f"Right mode mean {right_mode_mean} too far from 3"
        )

        # Statistical test: Compare against true mixture using Anderson-Darling
        # Generate reference samples from true distribution
        true_samples = []
        for _ in range(n_samples):
            if np.random.random() < 0.4:
                true_samples.append(np.random.normal(-2, 0.5))
            else:
                true_samples.append(np.random.normal(3, 0.8))
        true_samples = np.array(true_samples)

        # Two-sample KS test
        ks_stat, p_value = stats.ks_2samp(samples, true_samples)
        assert p_value > 0.01, (
            f"Samples don't match target distribution (p={p_value:.4f})"
        )

        print(
            f"Bimodal test - Acceptance rate: {stats_dict['acceptance_rate']:.3f}, KS p-value: {p_value:.3f}"
        )

    def test_heavy_tailed_distribution_with_outliers(self, random_seed):
        """
        Test Case 2: Heavy-tailed Student's t-distribution vs Normal Proposal

        Tests sampling from t-distribution with low degrees of freedom using
        normal proposal. This is challenging because:
        - Target has much heavier tails than proposal
        - High probability of extreme outliers
        - Tests robustness to numerical issues
        - Validates proper tail behavior
        """

        # Target: Student's t-distribution with df=2 (very heavy tails)
        df = 2

        def t_pdf(x):
            return stats.t.pdf(x, df=df)

        # Proposal: Standard normal (much lighter tails)
        # Proposal: Cauchy(0, 1) which has heavier tails than t(2)
        def cauchy_pdf(x):
            return stats.cauchy.pdf(x)

        def cauchy_sampler(n):
            return stats.cauchy.rvs(size=n)

        sampler = RejectionSampler(
            target_pdf=t_pdf,
            proposal_pdf=cauchy_pdf,
            proposal_sampler=cauchy_sampler,
            adaptive_M=True,
            initial_batch_size=5000,  # Larger batches due to low acceptance
        )

        n_samples = 8000
        samples = sampler.sample(n_samples)
        stats_dict = sampler.get_stats()

        # Basic validation
        assert len(samples) == n_samples, "Incorrect number of samples generated"
        assert stats_dict["acceptance_rate"] > 0.005, "Acceptance rate extremely low"

        # Heavy tail validation - check for outliers
        outliers_left = np.sum(samples < -3)
        outliers_right = np.sum(samples > 3)
        total_outliers = outliers_left + outliers_right

        # t(2) should have more outliers than normal distribution
        expected_normal_outliers = n_samples * (
            1 - stats.norm.cdf(3) + stats.norm.cdf(-3)
        )
        assert total_outliers > expected_normal_outliers * 2, (
            "Not enough outliers for heavy-tailed distribution"
        )

        # Test moments: t(2) has undefined variance, but sample variance should be large
        sample_var = np.var(samples)
        assert sample_var > 2.0, (
            f"Sample variance {sample_var} too small for t(2) distribution"
        )

        # Statistical validation using Kolmogorov-Smirnov test
        ks_stat, p_value = kstest(samples, lambda x: stats.t.cdf(x, df=df))
        assert p_value > 0.01, (
            f"KS test failed: samples don't follow t({df}) distribution (p={p_value:.4f})"
        )

        # Test extreme value behavior - should have some very large values
        max_abs_value = np.max(np.abs(samples))
        assert max_abs_value > 5.0, (
            f"Maximum absolute value {max_abs_value} too small for heavy-tailed distribution"
        )

        print(
            f"Heavy-tailed test - Acceptance rate: {stats_dict['acceptance_rate']:.4f}, "
            f"Sample variance: {sample_var:.2f}, KS p-value: {p_value:.3f}"
        )

    def test_multimodal_discrete_mixture_approximation(self, random_seed):
        """
        Test Case 3: Complex Multimodal Distribution with Edge Cases

        Tests a complex target distribution that's a mixture of different
        distribution types, including handling of:
        - Multiple modes at different scales
        - Bounded support regions
        - Discontinuities and edge cases
        - Numerical stability issues
        - Performance with challenging proposal
        """

        # Target: Complex mixture distribution
        # - 30% Beta(2, 5) scaled to [0, 2]
        # - 40% Exponential(rate=1) shifted to [3, ∞) and truncated at 8
        # - 30% Uniform on [10, 12]
        def complex_pdf(x):
            total = 0.0

            # Component 1: Scaled Beta(2,5) on [0, 2]
            if 0 <= x <= 2:
                # Beta(2,5) PDF is 30*u*(1-u)^4 where u = x/2
                u = x / 2
                beta_val = 30 * u * (1 - u) ** 4
                total += 0.3 * beta_val / 2  # Scale factor for [0,2]

            # Component 2: Truncated shifted exponential on [3, 8]
            if 3 <= x <= 8:
                shifted_x = x - 3
                exp_val = np.exp(-shifted_x)
                # Normalize by truncation factor
                normalization = 1 - np.exp(-5)  # 1 - exp(-5)
                total += 0.4 * exp_val / normalization

            # Component 3: Uniform on [10, 12]
            if 10 <= x <= 12:
                total += 0.3 / 2  # 0.3 probability over interval of length 2

            return total

        # Proposal: Mixture of uniforms covering all regions (poor fit intentionally)
        def proposal_pdf(x):
            if 0 <= x <= 12:
                return 1.0 / 12  # Uniform over [0, 12]
            return 0

        def proposal_sampler(n):
            return np.random.uniform(0, 12, n)

        sampler = RejectionSampler(
            target_pdf=complex_pdf,
            proposal_pdf=proposal_pdf,
            proposal_sampler=proposal_sampler,
            adaptive_M=True,
            initial_batch_size=3000,
        )

        n_samples = 12000
        samples = sampler.sample(n_samples)
        stats_dict = sampler.get_stats()

        # Basic validation
        assert len(samples) == n_samples, "Incorrect number of samples generated"
        assert stats_dict["acceptance_rate"] > 0.02, (
            "Acceptance rate too low for complex distribution"
        )

        # Validate support regions
        assert np.all((samples >= 0) & (samples <= 12)), (
            "Samples outside expected support"
        )

        # Check that samples appear in all three regions
        region1 = samples[(samples >= 0) & (samples <= 2)]
        region2 = samples[(samples >= 3) & (samples <= 8)]
        region3 = samples[(samples >= 10) & (samples <= 12)]

        # Should have samples in all regions (with some tolerance)
        assert len(region1) > 0.15 * n_samples, (
            f"Too few samples in region 1: {len(region1)}"
        )
        assert len(region2) > 0.25 * n_samples, (
            f"Too few samples in region 2: {len(region2)}"
        )
        assert len(region3) > 0.15 * n_samples, (
            f"Too few samples in region 3: {len(region3)}"
        )

        # Validate proportions approximately match target (30%, 40%, 30%)
        prop1 = len(region1) / n_samples
        prop2 = len(region2) / n_samples
        prop3 = len(region3) / n_samples

        assert abs(prop1 - 0.3) < 0.08, (
            f"Region 1 proportion {prop1:.3f} too far from 0.3"
        )
        assert abs(prop2 - 0.4) < 0.08, (
            f"Region 2 proportion {prop2:.3f} too far from 0.4"
        )
        assert abs(prop3 - 0.3) < 0.08, (
            f"Region 3 proportion {prop3:.3f} too far from 0.3"
        )

        # Test distribution properties within each region
        # Region 1: Should be left-skewed (beta with shape parameters 2,5)
        if len(region1) > 100:
            region1_skew = stats.skew(region1)
            assert region1_skew > 0.2, (
                f"Region 1 should be right-skewed, got skewness {region1_skew:.3f}"
            )

        # Region 2: Should be heavily right-skewed (exponential-like)
        if len(region2) > 100:
            region2_mean = np.mean(region2)
            assert region2_mean < 5.0, (
                f"Region 2 mean {region2_mean:.3f} should be closer to 3 (exponential start)"
            )

        # Region 3: Should be approximately uniform
        if len(region3) > 100:
            region3_var = np.var(region3)
            uniform_var = (12 - 10) ** 2 / 12  # Theoretical variance of Uniform(10,12)
            assert abs(region3_var - uniform_var) < 0.5, (
                f"Region 3 variance {region3_var:.3f} should be close to uniform variance {uniform_var:.3f}"
            )

        # Edge case validation: ensure no samples in gaps [2,3) and (8,10)
        gap1_samples = samples[(samples > 2) & (samples < 3)]
        gap2_samples = samples[(samples > 8) & (samples < 10)]

        assert len(gap1_samples) == 0, f"Found {len(gap1_samples)} samples in gap [2,3)"
        assert len(gap2_samples) == 0, (
            f"Found {len(gap2_samples)} samples in gap (8,10)"
        )

        print(
            f"Complex multimodal test - Acceptance rate: {stats_dict['acceptance_rate']:.3f}"
        )
        print(
            f"Proportions - Region 1: {prop1:.3f}, Region 2: {prop2:.3f}, Region 3: {prop3:.3f}"
        )

    def test_edge_cases_and_error_handling(self, random_seed):
        """Test edge cases, error conditions, and robustness."""

        # Test with invalid M (should raise error or handle gracefully)
        def simple_pdf(x):
            return 1 if 0 <= x <= 1 else 0

        def uniform_sampler(n):
            return np.random.uniform(0, 1, n)

        # Test M estimation failure
        def bad_proposal_pdf(x):
            return 0  # Always zero - should cause M estimation to fail

        def bad_proposal_sampler(n):
            return np.random.uniform(0, 1, n)

        with pytest.raises(ValueError, match="Could not estimate M"):
            sampler = RejectionSampler(
                target_pdf=simple_pdf,
                proposal_pdf=bad_proposal_pdf,
                proposal_sampler=bad_proposal_sampler,
                adaptive_M=True,
            )
            sampler.sample(100)

        # Test with very small sample size
        sampler = RejectionSampler(
            target_pdf=simple_pdf,
            proposal_pdf=simple_pdf,
            proposal_sampler=uniform_sampler,
            M=1.0,
        )

        small_samples = sampler.sample(1)
        assert len(small_samples) == 1, "Should handle single sample request"

        # Test with zero sample request
        zero_samples = sampler.sample(0)
        assert len(zero_samples) == 0, "Should handle zero sample request"
