from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


class AdaptiveRejectionSampler:
    """
    Adaptive Rejection Sampling for log-concave distributions.

    Builds piecewise linear envelopes that adaptively improve as sampling proceeds.
    """

    def __init__(
        self,
        log_pdf: Callable,
        domain: Tuple[float, float],
        initial_points: Optional[List[float]] = None,
    ):
        """
        Initialize adaptive rejection sampler.

        Args:
            log_pdf: Log of the probability density function
            domain: (lower_bound, upper_bound) of the distribution support
            initial_points: Initial points to build envelope (if None, uses domain endpoints)
        """
        self.log_pdf = log_pdf
        self.domain = domain
        self.envelope_points = []  # (x, log_f(x), log_f'(x))
        self.hull_points = []  # Points defining the piecewise linear envelope
        self.total_samples = 0
        self.accepted_samples = 0

        # Initialize envelope
        if initial_points is None:
            # Use domain endpoints and midpoint
            initial_points = [
                domain[0] + 0.01 * (domain[1] - domain[0]),
                (domain[0] + domain[1]) / 2,
                domain[1] - 0.01 * (domain[1] - domain[0]),
            ]

        self._initialize_envelope(initial_points)

    def _log_pdf_derivative(self, x: float, h: float = 1e-6) -> float:
        """Numerical derivative of log PDF."""
        try:
            return (self.log_pdf(x + h) - self.log_pdf(x - h)) / (2 * h)
        except Exception:
            return 0.0

    def _initialize_envelope(self, points: List[float]):
        """Initialize the piecewise linear envelope."""
        for x in sorted(points):
            if self.domain[0] <= x <= self.domain[1]:
                try:
                    log_f = self.log_pdf(x)
                    log_f_prime = self._log_pdf_derivative(x)
                    self.envelope_points.append((x, log_f, log_f_prime))
                except Exception:
                    continue

        self._update_hull()

    def _update_hull(self):
        """Update the piecewise linear hull (envelope)."""
        if len(self.envelope_points) < 2:
            return

        # Sort points by x-coordinate
        self.envelope_points.sort(key=lambda p: p[0])

        # Build hull segments
        self.hull_points = []
        for i in range(len(self.envelope_points)):
            x_i, log_f_i, log_f_prime_i = self.envelope_points[i]

            # Find intersection points with neighboring tangent lines
            left_intersect = self.domain[0]
            right_intersect = self.domain[1]

            if i > 0:
                x_prev, log_f_prev, log_f_prime_prev = self.envelope_points[i - 1]
                # Intersection: log_f_prev + log_f_prime_prev * (x - x_prev) = log_f_i + log_f_prime_i * (x - x_i)
                if abs(log_f_prime_i - log_f_prime_prev) > 1e-10:
                    intersect = (
                        log_f_i
                        - log_f_prev
                        - log_f_prime_i * x_i
                        + log_f_prime_prev * x_prev
                    ) / (log_f_prime_prev - log_f_prime_i)
                    left_intersect = max(left_intersect, intersect)

            if i < len(self.envelope_points) - 1:
                x_next, log_f_next, log_f_prime_next = self.envelope_points[i + 1]
                if abs(log_f_prime_next - log_f_prime_i) > 1e-10:
                    intersect = (
                        log_f_next
                        - log_f_i
                        - log_f_prime_next * x_next
                        + log_f_prime_i * x_i
                    ) / (log_f_prime_i - log_f_prime_next)
                    right_intersect = min(right_intersect, intersect)

            self.hull_points.append(
                (left_intersect, right_intersect, x_i, log_f_i, log_f_prime_i)
            )

    def _sample_from_envelope(self) -> float:
        """Sample from the piecewise linear envelope."""
        if not self.hull_points:
            return np.random.uniform(self.domain[0], self.domain[1])

        # Calculate areas under each segment
        areas = []
        for left, right, x_i, log_f_i, log_f_prime_i in self.hull_points:
            if left >= right:
                areas.append(0)
                continue

            # Area under exponential of linear function
            if abs(log_f_prime_i) < 1e-10:  # Constant segment
                area = (right - left) * np.exp(log_f_i)
            else:
                # ∫ exp(a + b*x) dx = exp(a)/b * (exp(b*right) - exp(b*left))
                a = log_f_i - log_f_prime_i * x_i
                b = log_f_prime_i
                try:
                    area = np.exp(a) / b * (np.exp(b * right) - np.exp(b * left))
                    if area < 0:
                        area = 0
                except Exception:
                    area = 0
            areas.append(area)

        total_area = sum(areas)
        if total_area <= 0:
            return np.random.uniform(self.domain[0], self.domain[1])

        # Select segment
        u = np.random.uniform(0, total_area)
        cumsum = 0
        for i, area in enumerate(areas):
            cumsum += area
            if u <= cumsum:
                # Sample from this segment
                left, right, x_i, log_f_i, log_f_prime_i = self.hull_points[i]

                if abs(log_f_prime_i) < 1e-10:
                    return np.random.uniform(left, right)

                # Inverse CDF sampling from exponential of linear function
                a = log_f_i - log_f_prime_i * x_i
                b = log_f_prime_i
                try:
                    F_left = np.exp(b * left)
                    F_right = np.exp(b * right)
                    u_segment = np.random.uniform(0, 1)
                    F_sample = F_left + u_segment * (F_right - F_left)
                    return np.log(F_sample) / b
                except Exception:
                    return np.random.uniform(left, right)

        return np.random.uniform(self.domain[0], self.domain[1])

    def _envelope_value(self, x: float) -> float:
        """Evaluate the envelope at point x."""
        for left, right, x_i, log_f_i, log_f_prime_i in self.hull_points:
            if left <= x <= right:
                return log_f_i + log_f_prime_i * (x - x_i)
        return -np.inf

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples using adaptive rejection sampling."""
        samples = []

        while len(samples) < n_samples:
            # Sample from envelope
            x = self._sample_from_envelope()
            self.total_samples += 1

            # Evaluate envelope and target at x
            log_envelope = self._envelope_value(x)
            log_target = self.log_pdf(x)

            # Acceptance test
            if np.log(np.random.uniform()) <= log_target - log_envelope:
                samples.append(x)
                self.accepted_samples += 1
            else:
                # Update envelope with rejected point
                try:
                    log_f_prime = self._log_pdf_derivative(x)
                    self.envelope_points.append((x, log_target, log_f_prime))
                    self._update_hull()
                except Exception:
                    pass

        return np.array(samples)

    def get_acceptance_rate(self) -> float:
        """Get current acceptance rate."""
        if self.total_samples == 0:
            return 0.0
        return self.accepted_samples / self.total_samples


class SqueezedRejectionSampler:
    """
    Squeezed Rejection Sampling with both upper envelope and lower squeeze function.

    Reduces expensive PDF evaluations by using squeeze function for quick acceptance.
    """

    def __init__(self, pdf: Callable, log_pdf: Callable, domain: Tuple[float, float]):
        self.pdf = pdf
        self.log_pdf = log_pdf
        self.domain = domain
        self.envelope_points = []
        self.squeeze_points = []
        self.pdf_evaluations = 0
        self.total_samples = 0
        self.squeeze_accepts = 0
        self.envelope_rejects = 0

        # Initialize with a few points
        initial_points = np.linspace(
            domain[0] + 0.1 * (domain[1] - domain[0]),
            domain[1] - 0.1 * (domain[1] - domain[0]),
            5,
        )
        self._initialize_functions(initial_points)

    def _initialize_functions(self, points: List[float]):
        """Initialize envelope and squeeze functions."""
        for x in points:
            try:
                log_f = self.log_pdf(x)
                log_f_prime = self._numerical_derivative(x)
                self.envelope_points.append((x, log_f, log_f_prime))
                self.pdf_evaluations += 1
            except Exception:
                continue

        self._update_functions()

    def _numerical_derivative(self, x: float, h: float = 1e-6) -> float:
        """Numerical derivative of log PDF."""
        try:
            return (self.log_pdf(x + h) - self.log_pdf(x - h)) / (2 * h)
        except Exception:
            return 0.0

    def _update_functions(self):
        """Update both envelope and squeeze functions."""
        if len(self.envelope_points) < 2:
            return

        # Sort points
        self.envelope_points.sort(key=lambda p: p[0])

        # Build squeeze function (piecewise linear lower bound)
        self.squeeze_points = []
        for i in range(len(self.envelope_points) - 1):
            x1, log_f1, _ = self.envelope_points[i]
            x2, log_f2, _ = self.envelope_points[i + 1]

            # Linear interpolation between consecutive points
            self.squeeze_points.append((x1, x2, log_f1, log_f2))

    def _envelope_value(self, x: float) -> float:
        """Evaluate envelope function at x."""
        # Find relevant envelope segment (similar to ARS)
        best_val = -np.inf
        for x_i, log_f_i, log_f_prime_i in self.envelope_points:
            val = log_f_i + log_f_prime_i * (x - x_i)
            best_val = max(best_val, val)
        return best_val

    def _squeeze_value(self, x: float) -> float:
        """Evaluate squeeze function at x."""
        for x1, x2, log_f1, log_f2 in self.squeeze_points:
            if x1 <= x <= x2:
                # Linear interpolation
                t = (x - x1) / (x2 - x1) if x2 > x1 else 0
                return log_f1 + t * (log_f2 - log_f1)
        return -np.inf

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples using squeezed rejection sampling."""
        samples = []

        while len(samples) < n_samples:
            # Sample from envelope (simplified uniform for demo)
            x = np.random.uniform(self.domain[0], self.domain[1])
            self.total_samples += 1

            log_envelope = self._envelope_value(x)
            log_squeeze = self._squeeze_value(x)

            u = np.log(np.random.uniform())

            # Quick acceptance using squeeze
            if u <= log_squeeze:
                samples.append(x)
                self.squeeze_accepts += 1
                continue

            # Quick rejection using envelope
            if u > log_envelope:
                self.envelope_rejects += 1
                continue

            # Need to evaluate actual PDF
            log_target = self.log_pdf(x)
            self.pdf_evaluations += 1

            if u <= log_target:
                samples.append(x)
                # Update functions with new point
                try:
                    log_f_prime = self._numerical_derivative(x)
                    self.envelope_points.append((x, log_target, log_f_prime))
                    if len(self.envelope_points) % 5 == 0:  # Update periodically
                        self._update_functions()
                except Exception:
                    pass

        return np.array(samples)

    def get_efficiency_stats(self) -> dict:
        """Get efficiency statistics."""
        return {
            "pdf_evaluations": self.pdf_evaluations,
            "total_samples": self.total_samples,
            "squeeze_acceptance_rate": self.squeeze_accepts
            / max(1, self.total_samples),
            "envelope_rejection_rate": self.envelope_rejects
            / max(1, self.total_samples),
            "pdf_evaluation_rate": self.pdf_evaluations / max(1, self.total_samples),
        }


class TransformedRejectionSampler:
    """
    Transformed Rejection Sampling using ratio-of-uniforms method.

    Transforms the sampling problem to make rejection sampling more efficient.
    """

    def __init__(self, pdf: Callable, mode: float = None):
        self.pdf = pdf
        self.mode = mode
        self.total_samples = 0
        self.accepted_samples = 0

        if mode is None:
            self.mode = self._find_mode()

        # Find bounding rectangle for ratio-of-uniforms
        self._find_bounding_rectangle()

    def _find_mode(self) -> float:
        """Find mode of the distribution."""
        try:
            result = minimize_scalar(
                lambda x: -self.pdf(x), bounds=(-10, 10), method="bounded"
            )
            return result.x
        except Exception:
            return 0.0

    def _find_bounding_rectangle(self):
        """Find bounding rectangle for ratio-of-uniforms transformation."""
        # For ratio-of-uniforms: u ~ U(0, sqrt(f(mode))), v ~ U(-c, c) where c depends on distribution
        self.u_max = np.sqrt(self.pdf(self.mode))

        # Find approximate bounds for v
        x_test = np.linspace(self.mode - 5, self.mode + 5, 1000)
        pdf_vals = np.array([self.pdf(x) for x in x_test])
        valid_mask = pdf_vals > 0

        if np.any(valid_mask):
            v_vals = x_test[valid_mask] * np.sqrt(pdf_vals[valid_mask])
            self.v_min = np.min(v_vals) * 1.1
            self.v_max = np.max(v_vals) * 1.1
        else:
            self.v_min, self.v_max = -5, 5

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples using ratio-of-uniforms transformation."""
        samples = []

        while len(samples) < n_samples:
            # Sample from bounding rectangle
            u = np.random.uniform(0, self.u_max)
            v = np.random.uniform(self.v_min, self.v_max)

            self.total_samples += 1

            # Transform back to original space
            if u > 0:
                x = v / u

                # Acceptance test: u² ≤ f(x)
                if u * u <= self.pdf(x):
                    samples.append(x)
                    self.accepted_samples += 1

        return np.array(samples)

    def get_acceptance_rate(self) -> float:
        return self.accepted_samples / max(1, self.total_samples)
