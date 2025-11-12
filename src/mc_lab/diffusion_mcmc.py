"""Diffusion-model assisted Markov Chain Monte Carlo sampling.

This module implements the algorithm from Hunt-Smith et al. (2023):
"Accelerating Markov Chain Monte Carlo sampling with diffusion models"
arXiv:2309.01454v1

The key idea is to augment a Metropolis-Hastings sampler with a diffusion model
that learns to approximate the posterior distribution during sampling. This allows
for non-local jumps between modes while maintaining detailed balance.
"""

from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

from mc_lab._rng import RandomState, as_generator


class DiffusionModel:
    """A simplified diffusion model for low-dimensional MCMC samples.

    This model uses a forward diffusion process to add Gaussian noise progressively,
    then learns a reverse process by fitting parameters that minimize reconstruction error.
    Unlike image-based diffusion models that use neural networks, this implementation
    fits simple scalar parameters for efficiency.

    Args:
        n_steps: Number of diffusion time steps.
        beta_min: Minimum noise variance (at t=1).
        beta_max: Maximum noise variance (at t=T).
        random_state: Random number generator seed or instance.

    Attributes:
        n_steps: Number of diffusion time steps.
        betas: Variance schedule for noise addition.
        phi: Learned parameters for the reverse process (shape: [n_steps, n_dims]).
        is_trained: Whether the model has been trained.
    """

    def __init__(
        self,
        n_steps: int = 20,
        beta_min: float = 0.1,
        beta_max: float = 0.3,
        random_state: RandomState = None,
    ):
        self.n_steps = n_steps
        self.betas = np.linspace(beta_min, beta_max, n_steps)
        self.rng = as_generator(random_state)
        self.phi: Optional[np.ndarray] = None
        self.is_trained = False
        self.n_dims = 0

    def forward_diffusion(self, samples: np.ndarray) -> list[np.ndarray]:
        """Apply forward diffusion process to samples.

        Progressively adds Gaussian noise according to:
        x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon

        Args:
            samples: Array of shape (n_samples, n_dims).

        Returns:
            List of arrays at each time step, from x_0 to x_T.
        """
        trajectory = [samples.copy()]

        for t in range(self.n_steps):
            x_prev = trajectory[-1]
            epsilon = self.rng.standard_normal(x_prev.shape)
            x_t = np.sqrt(1 - self.betas[t]) * x_prev + np.sqrt(self.betas[t]) * epsilon
            trajectory.append(x_t)

        return trajectory

    def train(self, samples: np.ndarray, method: str = "L-BFGS-B") -> None:
        """Train the diffusion model on MCMC samples.

        Fits parameters phi_t for the reverse process by minimizing:
        L = sum_t |y_{t-1}(phi_t) - x_{t-1}|^2

        where y_{t-1} = y_t - (x_t - x_{t-1}) * phi_t

        Args:
            samples: Training samples of shape (n_samples, n_dims).
            method: Optimization method for scipy.minimize.
        """
        if samples.shape[0] == 0:
            raise ValueError("Cannot train on empty sample set")

        self.n_dims = samples.shape[1]

        # Apply forward diffusion
        forward_trajectory = self.forward_diffusion(samples)

        # Initialize parameters
        n_params = self.n_steps * self.n_dims
        phi_init = self.rng.normal(0, 0.1, n_params)

        def loss(phi_flat: np.ndarray) -> float:
            """Loss function for optimization."""
            phi = phi_flat.reshape(self.n_steps, self.n_dims)

            # Start from pure noise
            y_t = self.rng.standard_normal(samples.shape)
            total_loss = 0.0

            # Apply reverse process
            for t in range(self.n_steps - 1, -1, -1):
                x_t = forward_trajectory[t + 1]
                x_prev = forward_trajectory[t]
                diff = x_t - x_prev

                y_prev = y_t - diff * phi[t]

                # Accumulate squared error
                total_loss += np.sum((y_prev - x_prev) ** 2)

                y_t = y_prev

            return total_loss

        # Optimize
        result = minimize(loss, phi_init, method=method)

        if not result.success:
            # Still use the result even if optimization didn't fully converge
            pass

        self.phi = result.x.reshape(self.n_steps, self.n_dims)
        self.is_trained = True

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples using the trained reverse diffusion process.

        Starts from pure Gaussian noise and applies the learned reverse transformation.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Array of shape (n_samples, n_dims).

        Raises:
            RuntimeError: If model has not been trained.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before sampling")

        # Start from pure noise
        z_t = self.rng.standard_normal((n_samples, self.n_dims))

        # Apply learned reverse process
        # We need to store the forward diffs, but we'll approximate them
        # For sampling, we use the trained phi directly
        for t in range(self.n_steps - 1, -1, -1):
            # Approximate the forward diff using the reverse process
            # This is a simplified version - in practice, we sample from noise each time
            epsilon = self.rng.standard_normal((n_samples, self.n_dims))
            diff_approx = np.sqrt(self.betas[t]) * epsilon
            z_t = z_t - diff_approx * self.phi[t]

        return z_t


def estimate_proposal_probability(
    theta: np.ndarray,
    diffusion_model: DiffusionModel,
    n_samples_for_q: int = 1000,
    n_bins: int = 20,
) -> float:
    """Estimate Q(theta) using Gibbs sampling with 1D histograms.

    Computes Q(theta) = Q(theta_1) * Q(theta_2|theta_1) * ... using
    marginal and conditional distributions estimated from diffusion samples.

    Args:
        theta: Parameter vector to evaluate.
        diffusion_model: Trained diffusion model.
        n_samples_for_q: Number of diffusion samples for Q estimation.
        n_bins: Number of bins for 1D histograms.

    Returns:
        Estimated probability Q(theta).
    """
    n_dims = len(theta)
    q_prob = 1.0

    # Generate samples from diffusion model
    samples = diffusion_model.sample(n_samples_for_q)

    for d in range(n_dims):
        # Condition on previous dimensions
        if d > 0:
            # Filter samples where previous dimensions match (within tolerance)
            mask = np.ones(len(samples), dtype=bool)
            for prev_d in range(d):
                # Simple conditioning: keep samples close to theta[prev_d]
                sample_range = samples[:, prev_d].max() - samples[:, prev_d].min()
                tolerance = sample_range / n_bins
                mask &= np.abs(samples[:, prev_d] - theta[prev_d]) < tolerance

            if np.sum(mask) < 5:  # Not enough samples
                return 1e-10  # Small probability

            samples = samples[mask]

        # Estimate marginal/conditional probability for this dimension
        values = samples[:, d]

        if len(values) == 0:
            return 1e-10

        # Create histogram
        counts, bin_edges = np.histogram(values, bins=n_bins)
        counts = counts + 1  # Laplace smoothing

        # Find which bin theta[d] falls into
        bin_idx = np.searchsorted(bin_edges[:-1], theta[d], side="right") - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        # Probability is proportional to count in bin
        q_prob *= counts[bin_idx] / np.sum(counts)

        if q_prob < 1e-300:  # Numerical underflow protection
            return 1e-10

    return max(q_prob, 1e-10)  # Avoid exact zero


def diffusion_assisted_mcmc(
    log_posterior: Callable[[np.ndarray], float],
    initial_theta: np.ndarray,
    n_samples: int,
    p_diff: float = 0.5,
    sigma_mh: float = 0.1,
    retrain_interval: int = 100,
    n_diffusion_steps: int = 20,
    beta_min: float = 0.1,
    beta_max: float = 0.3,
    n_samples_for_q: int = 1000,
    n_bins: int = 20,
    seed_samples: Optional[np.ndarray] = None,
    random_state: RandomState = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Diffusion-model assisted Metropolis-Hastings MCMC sampling (Algorithm 1).

    This implements the algorithm from Hunt-Smith et al. (2023). The sampler alternates
    between local Gaussian proposals (standard MH) and global proposals from a diffusion
    model that is periodically retrained on collected samples.

    Args:
        log_posterior: Function that computes log P(theta|D). Higher is better.
        initial_theta: Starting point in parameter space.
        n_samples: Total number of MCMC samples to generate.
        p_diff: Probability of using diffusion proposal vs Gaussian proposal.
        sigma_mh: Standard deviation for Gaussian proposals.
        retrain_interval: Number of samples between diffusion model retraining.
        n_diffusion_steps: Number of time steps in diffusion process.
        beta_min: Minimum noise variance in diffusion.
        beta_max: Maximum noise variance in diffusion.
        n_samples_for_q: Number of diffusion samples for Q(theta) estimation.
        n_bins: Number of bins for probability estimation histograms.
        seed_samples: Optional initial samples to seed the diffusion model
            (shape: [n_seed, n_dims]). Useful for multi-modal distributions.
        random_state: Random number generator seed or instance.
        verbose: Whether to print progress information.

    Returns:
        samples: Array of samples, shape (n_samples, n_dims).
        accepted: Boolean array indicating which proposals were accepted.
        info: Dictionary with diagnostic information including:
            - n_diffusion_proposals: Number of diffusion proposals made
            - n_diffusion_accepted: Number of diffusion proposals accepted
            - n_gaussian_proposals: Number of Gaussian proposals made
            - n_gaussian_accepted: Number of Gaussian proposals accepted
            - diffusion_acceptance_rate: Acceptance rate for diffusion proposals
            - gaussian_acceptance_rate: Acceptance rate for Gaussian proposals

    Example:
        >>> # Sample from a 2D Gaussian
        >>> def log_posterior(theta):
        ...     return -0.5 * np.sum(theta**2)
        >>> initial = np.array([0.0, 0.0])
        >>> samples, accepted, info = diffusion_assisted_mcmc(
        ...     log_posterior, initial, n_samples=1000
        ... )
        >>> print(f"Overall acceptance rate: {np.mean(accepted):.2%}")
    """
    rng = as_generator(random_state)
    n_dims = len(initial_theta)

    # Initialize storage
    samples = np.zeros((n_samples, n_dims))
    accepted = np.zeros(n_samples, dtype=bool)
    samples[0] = initial_theta
    current_theta = initial_theta.copy()
    current_log_p = log_posterior(current_theta)

    # Initialize diffusion model
    diffusion_model = DiffusionModel(
        n_steps=n_diffusion_steps,
        beta_min=beta_min,
        beta_max=beta_max,
        random_state=rng,
    )

    # Seed the model if initial samples provided
    if seed_samples is not None:
        all_samples = seed_samples.copy()
    else:
        all_samples = initial_theta.reshape(1, -1).copy()

    # Statistics
    n_diff_proposed = 0
    n_diff_accepted = 0
    n_gauss_proposed = 0
    n_gauss_accepted = 0

    for i in range(1, n_samples):
        if verbose and i % 1000 == 0:
            print(f"Sample {i}/{n_samples}")

        # Retrain diffusion model periodically
        if i % retrain_interval == 0 and i > 0:
            if verbose:
                print(f"  Retraining diffusion model on {len(all_samples)} samples...")
            try:
                diffusion_model.train(all_samples)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Diffusion training failed: {e}")

        # Decide proposal type
        use_diffusion = rng.random() < p_diff and diffusion_model.is_trained

        if use_diffusion:
            # Diffusion proposal (independence MH)
            n_diff_proposed += 1
            proposal = diffusion_model.sample(1)[0]
            proposal_log_p = log_posterior(proposal)

            # Estimate Q(theta) and Q(proposal) for acceptance ratio
            q_current = estimate_proposal_probability(
                current_theta, diffusion_model, n_samples_for_q, n_bins
            )
            q_proposal = estimate_proposal_probability(
                proposal, diffusion_model, n_samples_for_q, n_bins
            )

            # Acceptance probability: min(1, P(proposal)/P(current) * Q(current)/Q(proposal))
            log_ratio = proposal_log_p - current_log_p + np.log(q_current) - np.log(
                q_proposal
            )
            accept_prob = min(1.0, np.exp(log_ratio))

        else:
            # Gaussian proposal (standard MH)
            n_gauss_proposed += 1
            proposal = current_theta + sigma_mh * rng.standard_normal(n_dims)
            proposal_log_p = log_posterior(proposal)

            # Symmetric proposal, so Q terms cancel
            log_ratio = proposal_log_p - current_log_p
            accept_prob = min(1.0, np.exp(log_ratio))

        # Accept/reject
        if rng.random() < accept_prob:
            current_theta = proposal
            current_log_p = proposal_log_p
            accepted[i] = True

            if use_diffusion:
                n_diff_accepted += 1
            else:
                n_gauss_accepted += 1

        samples[i] = current_theta
        all_samples = np.vstack([all_samples, current_theta])

    # Compute statistics
    diff_accept_rate = n_diff_accepted / n_diff_proposed if n_diff_proposed > 0 else 0.0
    gauss_accept_rate = (
        n_gauss_accepted / n_gauss_proposed if n_gauss_proposed > 0 else 0.0
    )

    info = {
        "n_diffusion_proposals": n_diff_proposed,
        "n_diffusion_accepted": n_diff_accepted,
        "n_gaussian_proposals": n_gauss_proposed,
        "n_gaussian_accepted": n_gauss_accepted,
        "diffusion_acceptance_rate": diff_accept_rate,
        "gaussian_acceptance_rate": gauss_accept_rate,
    }

    if verbose:
        print(f"\nSampling complete!")
        print(f"Diffusion proposals: {n_diff_proposed}, accepted: {n_diff_accepted}")
        print(f"  Acceptance rate: {diff_accept_rate:.2%}")
        print(f"Gaussian proposals: {n_gauss_proposed}, accepted: {n_gauss_accepted}")
        print(f"  Acceptance rate: {gauss_accept_rate:.2%}")
        print(f"Overall acceptance rate: {np.mean(accepted):.2%}")

    return samples, accepted, info
