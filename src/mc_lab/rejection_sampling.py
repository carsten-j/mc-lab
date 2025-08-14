import warnings
from typing import Callable

import numpy as np

# Import the RejectionSampler class (assuming it's in rejection_sampling.py)
# from rejection_sampling import RejectionSampler


# For testing purposes, I'll include a minimal version of the class here
class RejectionSampler:
    """Fast and efficient rejection sampling implementation."""

    def __init__(
        self,
        target_pdf: Callable,
        proposal_pdf: Callable,
        proposal_sampler: Callable,
        M: float = None,
        adaptive_M: bool = True,
        initial_batch_size: int = 1000,
    ):
        self.target_pdf = target_pdf
        self.proposal_pdf = proposal_pdf
        self.proposal_sampler = proposal_sampler
        self.M = M
        self.adaptive_M = adaptive_M
        self.batch_size = initial_batch_size
        self.acceptance_rate = 0.0
        self.total_proposals = 0
        self.total_accepted = 0

    def _estimate_M(self, n_samples: int = 10000) -> float:
        """Estimate the bound M by sampling from the proposal distribution."""
        proposals = self.proposal_sampler(n_samples)
        if np.isscalar(proposals):
            proposals = np.array([proposals])
        elif not isinstance(proposals, np.ndarray):
            proposals = np.array(proposals)

        if proposals.ndim == 0:
            proposals = proposals.reshape(1)

        ratios = []
        for x in proposals:
            try:
                target_val = self.target_pdf(x)
                proposal_val = self.proposal_pdf(x)
                if proposal_val > 0:
                    ratios.append(target_val / proposal_val)
            except Exception:
                continue

        if not ratios:
            raise ValueError("Could not estimate M - check your distributions")

        M_est = np.percentile(ratios, 99.9) * 1.1
        return max(M_est, 1.0)

    def _adaptive_batch_size(self):
        """Adaptively adjust batch size based on acceptance rate."""
        if self.total_proposals > 100:
            target_acceptance = 0.2
            if self.acceptance_rate < target_acceptance / 2:
                self.batch_size = min(self.batch_size * 2, 50000)
            elif self.acceptance_rate > target_acceptance * 2:
                self.batch_size = max(self.batch_size // 2, 100)

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples using rejection sampling."""
        if n_samples <= 0:
            return np.array([], dtype=float)
        if self.M is None or self.adaptive_M:
            self.M = self._estimate_M()

        samples = []
        remaining = n_samples

        while remaining > 0:
            batch_size = min(self.batch_size, remaining * 5)
            proposals = self.proposal_sampler(batch_size)

            if np.isscalar(proposals):
                proposals = np.array([proposals])
            elif not isinstance(proposals, np.ndarray):
                proposals = np.array(proposals)

            if proposals.ndim == 0:
                proposals = proposals.reshape(1)

            try:
                target_vals = np.array(
                    [self.target_pdf(x) for x in proposals], dtype=float
                )
                proposal_vals = np.array(
                    [self.proposal_pdf(x) for x in proposals], dtype=float
                )
            except Exception:
                target_vals = []
                proposal_vals = []
                for x in proposals:
                    try:
                        target_vals.append(self.target_pdf(x))
                        proposal_vals.append(self.proposal_pdf(x))
                    except Exception:
                        target_vals.append(0)
                        proposal_vals.append(1)
                target_vals = np.array(target_vals, dtype=float)
                proposal_vals = np.array(proposal_vals, dtype=float)

            valid_mask = proposal_vals > 0
            # Ensure floating point probabilities to avoid integer truncation
            acceptance_probs = np.zeros(target_vals.shape, dtype=float)
            acceptance_probs[valid_mask] = target_vals[valid_mask] / (
                self.M * proposal_vals[valid_mask]
            )
            # Numerical safety: clip to [0, 1]
            np.clip(acceptance_probs, 0.0, 1.0, out=acceptance_probs)

            u = np.random.uniform(0, 1, len(proposals))
            accepted_mask = (u <= acceptance_probs) & valid_mask

            accepted_samples = proposals[accepted_mask]
            n_accepted = len(accepted_samples)

            if n_accepted > 0:
                take = min(n_accepted, remaining)
                samples.extend(accepted_samples[:take])
                remaining -= take

            self.total_proposals += len(proposals)
            self.total_accepted += n_accepted
            if self.total_proposals > 0:
                self.acceptance_rate = self.total_accepted / self.total_proposals

            self._adaptive_batch_size()

            if self.acceptance_rate < 1e-6 and self.total_proposals > 100000:
                warnings.warn("Very low acceptance rate. Check if M is appropriate.")
                break

        return np.array(samples[:n_samples])

    def get_stats(self) -> dict:
        """Return sampling statistics."""
        return {
            "acceptance_rate": self.acceptance_rate,
            "total_proposals": self.total_proposals,
            "total_accepted": self.total_accepted,
            "current_M": self.M,
            "current_batch_size": self.batch_size,
        }
