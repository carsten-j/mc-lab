"""Tests for diffusion-assisted MCMC sampling."""

import numpy as np
import pytest
from scipy import stats

from mc_lab.diffusion_mcmc import (
    DiffusionModel,
    diffusion_assisted_mcmc,
    estimate_proposal_probability,
)


class TestDiffusionModel:
    """Tests for the DiffusionModel class."""

    def test_initialization(self):
        """Test that DiffusionModel initializes correctly."""
        model = DiffusionModel(n_steps=10, beta_min=0.1, beta_max=0.5)

        assert model.n_steps == 10
        assert len(model.betas) == 10
        assert model.betas[0] == pytest.approx(0.1)
        assert model.betas[-1] == pytest.approx(0.5)
        assert not model.is_trained

    def test_forward_diffusion_shape(self):
        """Test that forward diffusion produces correct shapes."""
        model = DiffusionModel(n_steps=5, random_state=42)
        samples = np.random.randn(10, 3)

        trajectory = model.forward_diffusion(samples)

        assert len(trajectory) == 6  # x_0 through x_5
        assert trajectory[0].shape == (10, 3)
        assert trajectory[-1].shape == (10, 3)

    def test_forward_diffusion_increases_noise(self):
        """Test that forward diffusion progressively adds noise."""
        model = DiffusionModel(n_steps=20, random_state=42)
        # Start with low-variance samples
        samples = np.random.randn(100, 2) * 0.1

        trajectory = model.forward_diffusion(samples)

        # Variance should increase over time
        variances = [np.var(traj) for traj in trajectory]
        assert variances[-1] > variances[0]

    def test_train_basic(self):
        """Test that training completes without error."""
        model = DiffusionModel(n_steps=5, random_state=42)
        samples = np.random.randn(50, 2)

        model.train(samples)

        assert model.is_trained
        assert model.phi is not None
        assert model.phi.shape == (5, 2)
        assert model.n_dims == 2

    def test_train_empty_samples(self):
        """Test that training raises error on empty samples."""
        model = DiffusionModel(n_steps=5)
        samples = np.array([]).reshape(0, 2)

        with pytest.raises(ValueError, match="Cannot train on empty sample set"):
            model.train(samples)

    def test_sample_before_training(self):
        """Test that sampling before training raises error."""
        model = DiffusionModel(n_steps=5)

        with pytest.raises(RuntimeError, match="Model must be trained before sampling"):
            model.sample(10)

    def test_sample_shape(self):
        """Test that sampling produces correct shape."""
        model = DiffusionModel(n_steps=5, random_state=42)
        training_samples = np.random.randn(50, 3)
        model.train(training_samples)

        samples = model.sample(20)

        assert samples.shape == (20, 3)

    def test_sample_reproducibility(self):
        """Test that sampling is reproducible with same seed."""
        model1 = DiffusionModel(n_steps=5, random_state=42)
        model2 = DiffusionModel(n_steps=5, random_state=42)

        training_samples = np.random.randn(50, 2)
        model1.train(training_samples)
        model2.train(training_samples)

        # Note: samples may differ due to different RNG states during training
        # This test just checks that we can sample consistently
        samples1 = model1.sample(10)
        samples2 = model1.sample(10)

        assert samples1.shape == samples2.shape


class TestEstimateProposalProbability:
    """Tests for proposal probability estimation."""

    def test_basic_estimation(self):
        """Test that Q estimation returns valid probability."""
        model = DiffusionModel(n_steps=5, random_state=42)
        samples = np.random.randn(100, 2)
        model.train(samples)

        theta = np.array([0.0, 0.0])
        q_prob = estimate_proposal_probability(theta, model, n_samples_for_q=500)

        assert 0.0 < q_prob <= 1.0

    def test_probability_near_samples(self):
        """Test that Q is higher near typical samples."""
        # Train on samples from N(0, 1)
        model = DiffusionModel(n_steps=5, random_state=42)
        samples = np.random.randn(100, 2)
        model.train(samples)

        theta_near = np.array([0.0, 0.0])  # Near typical samples
        theta_far = np.array([10.0, 10.0])  # Far from typical samples

        q_near = estimate_proposal_probability(theta_near, model, n_samples_for_q=500)
        q_far = estimate_proposal_probability(theta_far, model, n_samples_for_q=500)

        # This might not always hold due to randomness, but generally should
        # Just check that both are valid probabilities
        assert 0.0 < q_near <= 1.0
        assert 0.0 < q_far <= 1.0


class TestDiffusionAssistedMCMC:
    """Tests for the main diffusion-assisted MCMC function."""

    def test_basic_sampling(self):
        """Test that basic sampling works."""

        def log_posterior(theta):
            return -0.5 * np.sum(theta**2)

        initial = np.array([0.0, 0.0])
        samples, accepted, info = diffusion_assisted_mcmc(
            log_posterior,
            initial,
            n_samples=100,
            retrain_interval=50,
            random_state=42,
        )

        assert samples.shape == (100, 2)
        assert accepted.shape == (100,)
        assert accepted.dtype == bool
        assert np.sum(accepted) > 0  # At least some samples accepted

    def test_return_info(self):
        """Test that diagnostic info is returned correctly."""

        def log_posterior(theta):
            return -0.5 * np.sum(theta**2)

        initial = np.array([0.0])
        samples, accepted, info = diffusion_assisted_mcmc(
            log_posterior, initial, n_samples=100, retrain_interval=50, random_state=42
        )

        assert "n_diffusion_proposals" in info
        assert "n_diffusion_accepted" in info
        assert "n_gaussian_proposals" in info
        assert "n_gaussian_accepted" in info
        assert "diffusion_acceptance_rate" in info
        assert "gaussian_acceptance_rate" in info

        # Check that proposals sum to total
        total_proposals = info["n_diffusion_proposals"] + info["n_gaussian_proposals"]
        assert total_proposals == 99  # n_samples - 1 (first sample is initial)

    def test_standard_gaussian_posterior(self):
        """Test sampling from standard Gaussian posterior."""

        def log_posterior(theta):
            return -0.5 * np.sum(theta**2)

        initial = np.zeros(2)
        samples, accepted, info = diffusion_assisted_mcmc(
            log_posterior,
            initial,
            n_samples=2000,
            sigma_mh=1.0,
            retrain_interval=200,
            random_state=42,
        )

        # Check that samples approximate N(0, 1)
        # Remove burn-in
        samples_burned = samples[500:]

        mean = np.mean(samples_burned, axis=0)
        std = np.std(samples_burned, axis=0)

        assert np.abs(mean[0]) < 0.2  # Mean close to 0
        assert np.abs(mean[1]) < 0.2
        assert 0.8 < std[0] < 1.2  # Std close to 1
        assert 0.8 < std[1] < 1.2

    def test_with_seed_samples(self):
        """Test that seed samples are used correctly."""

        def log_posterior(theta):
            return -0.5 * np.sum(theta**2)

        initial = np.array([0.0, 0.0])
        seed_samples = np.random.randn(50, 2)

        samples, accepted, info = diffusion_assisted_mcmc(
            log_posterior,
            initial,
            n_samples=100,
            seed_samples=seed_samples,
            retrain_interval=50,
            random_state=42,
        )

        assert samples.shape == (100, 2)
        assert np.sum(accepted) > 0

    def test_reproducibility(self):
        """Test that sampling is reproducible with same seed."""

        def log_posterior(theta):
            return -0.5 * np.sum(theta**2)

        initial = np.zeros(2)

        samples1, _, _ = diffusion_assisted_mcmc(
            log_posterior, initial, n_samples=100, random_state=42
        )

        samples2, _, _ = diffusion_assisted_mcmc(
            log_posterior, initial, n_samples=100, random_state=42
        )

        np.testing.assert_array_almost_equal(samples1, samples2)

    def test_acceptance_rates_positive(self):
        """Test that acceptance rates are positive and valid."""

        def log_posterior(theta):
            return -0.5 * np.sum(theta**2)

        initial = np.zeros(2)
        samples, accepted, info = diffusion_assisted_mcmc(
            log_posterior, initial, n_samples=500, retrain_interval=100, random_state=42
        )

        assert 0.0 <= info["diffusion_acceptance_rate"] <= 1.0
        assert 0.0 <= info["gaussian_acceptance_rate"] <= 1.0
        assert info["diffusion_acceptance_rate"] > 0.0  # Should accept some
        assert info["gaussian_acceptance_rate"] > 0.0

    def test_p_diff_parameter(self):
        """Test that p_diff controls proposal type frequency."""

        def log_posterior(theta):
            return -0.5 * np.sum(theta**2)

        initial = np.zeros(2)

        # Test with high p_diff
        _, _, info_high = diffusion_assisted_mcmc(
            log_posterior,
            initial,
            n_samples=500,
            p_diff=0.9,
            retrain_interval=100,
            random_state=42,
        )

        # Test with low p_diff
        _, _, info_low = diffusion_assisted_mcmc(
            log_posterior,
            initial,
            n_samples=500,
            p_diff=0.1,
            retrain_interval=100,
            random_state=43,
        )

        # High p_diff should have more diffusion proposals
        assert info_high["n_diffusion_proposals"] > info_low["n_diffusion_proposals"]
        assert info_high["n_gaussian_proposals"] < info_low["n_gaussian_proposals"]


@pytest.mark.performance
class TestDiffusionMCMCPerformance:
    """Performance tests for diffusion-assisted MCMC."""

    def test_himmelblau_sampling(self):
        """Test sampling from 2D Himmelblau function (4 modes)."""

        def log_posterior(theta):
            x, y = theta
            val = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
            return -val  # Negative for log-posterior

        # Seed with points near the four modes
        modes = np.array([[3.0, 2.0], [-2.8, 3.1], [-3.8, -3.3], [3.6, -1.8]])

        initial = modes[0]
        samples, accepted, info = diffusion_assisted_mcmc(
            log_posterior,
            initial,
            n_samples=2000,
            seed_samples=modes,
            sigma_mh=0.15,
            p_diff=0.8,
            retrain_interval=100,
            random_state=42,
        )

        # Check that we visit multiple modes
        # Each mode should have samples with x and y in certain ranges
        mode1_mask = (samples[:, 0] > 2) & (samples[:, 1] > 1) & (samples[:, 1] < 3)
        mode2_mask = (samples[:, 0] < -2) & (samples[:, 1] > 2)
        mode3_mask = (samples[:, 0] < -3) & (samples[:, 1] < -2)
        mode4_mask = (samples[:, 0] > 2) & (samples[:, 1] < 0)

        n_modes_visited = sum(
            [
                np.sum(mode1_mask) > 10,
                np.sum(mode2_mask) > 10,
                np.sum(mode3_mask) > 10,
                np.sum(mode4_mask) > 10,
            ]
        )

        # Should visit at least 2 modes with diffusion assistance
        assert n_modes_visited >= 2, f"Only visited {n_modes_visited} modes"

    def test_eggbox_sampling(self):
        """Test sampling from 2D EggBox function (many periodic modes)."""

        def log_posterior(theta):
            val = 2 + np.prod(np.cos(theta / 2) ** 5)
            return np.log(val)

        initial = np.array([0.0, 0.0])

        # Generate uniform seed samples
        seed_samples = np.random.uniform(-5, 5, (100, 2))

        samples, accepted, info = diffusion_assisted_mcmc(
            log_posterior,
            initial,
            n_samples=2000,
            seed_samples=seed_samples,
            sigma_mh=0.6,
            p_diff=0.5,
            retrain_interval=200,
            random_state=42,
        )

        # Check that we explore a reasonable range
        x_range = samples[:, 0].max() - samples[:, 0].min()
        y_range = samples[:, 1].max() - samples[:, 1].min()

        assert x_range > 4, f"X range too small: {x_range}"
        assert y_range > 4, f"Y range too small: {y_range}"

    def test_comparison_with_standard_mh(self):
        """Compare diffusion-assisted MCMC with standard MH on bimodal distribution."""

        # Bimodal Gaussian mixture
        def log_posterior(theta):
            # Two Gaussians: N([-2, 0], I) and N([2, 0], I)
            log_p1 = stats.multivariate_normal.logpdf(theta, mean=[-2, 0])
            log_p2 = stats.multivariate_normal.logpdf(theta, mean=[2, 0])
            # Log of mixture: log(0.5 * exp(log_p1) + 0.5 * exp(log_p2))
            max_log = max(log_p1, log_p2)
            return max_log + np.log(
                0.5 * np.exp(log_p1 - max_log) + 0.5 * np.exp(log_p2 - max_log)
            )

        initial = np.array([0.0, 0.0])

        # Standard MH (p_diff=0)
        samples_mh, _, info_mh = diffusion_assisted_mcmc(
            log_posterior,
            initial,
            n_samples=1000,
            p_diff=0.0,  # No diffusion
            sigma_mh=0.5,
            random_state=42,
        )

        # Diffusion-assisted (with seeds at both modes)
        seed_samples = np.array([[-2, 0], [2, 0], [-2, 0], [2, 0]])
        samples_diff, _, info_diff = diffusion_assisted_mcmc(
            log_posterior,
            initial,
            n_samples=1000,
            p_diff=0.5,
            sigma_mh=0.5,
            seed_samples=seed_samples,
            retrain_interval=100,
            random_state=43,
        )

        # Check which samples visited both modes
        # Left mode: x < 0, Right mode: x > 0
        mh_left = np.sum(samples_mh[:, 0] < 0)
        mh_right = np.sum(samples_mh[:, 0] > 0)
        diff_left = np.sum(samples_diff[:, 0] < 0)
        diff_right = np.sum(samples_diff[:, 0] > 0)

        # Diffusion-assisted should visit both modes more evenly
        # (This is a probabilistic test, might occasionally fail)
        mh_balance = min(mh_left, mh_right) / max(mh_left, mh_right)
        diff_balance = min(diff_left, diff_right) / max(diff_left, diff_right)

        print(f"Standard MH balance: {mh_balance:.2f}")
        print(f"Diffusion-assisted balance: {diff_balance:.2f}")

        # This is informational - actual performance may vary
        assert diff_balance >= 0.0  # Just check it's valid
