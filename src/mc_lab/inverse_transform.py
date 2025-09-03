from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq


class BaseInverseTransformSampler(ABC):
    """
    Base class for all inverse transform sampling methods.
    Provides unified interface for both analytical and numerical inverse CDFs.
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    @abstractmethod
    def _get_inverse_cdf(
        self,
    ) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
        """Return the inverse CDF function (analytical or numerical)."""
        pass

    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate n samples using inverse transform method.

        Parameters:
        -----------
        n : int
            Number of samples to generate.
        random_state : int, optional
            Random seed. If provided, overrides instance random_state.

        Returns:
        --------
        np.ndarray
            Array of n samples from the distribution.
        """

        # Set random seed
        seed = random_state if random_state is not None else self.random_state
        if seed is not None:
            np.random.seed(seed)

        # Generate uniform random variables
        u = self._generate_uniform(n)

        # Apply inverse CDF
        inverse_cdf = self._get_inverse_cdf()
        samples = inverse_cdf(u)

        return samples

    def _generate_uniform(self, n: int) -> np.ndarray:
        """
        Generate uniform random variables. Override for quasi-random,
        stratified, etc.
        """
        return np.random.uniform(0, 1, n)

    def sample_quantiles(
        self, quantiles: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Generate samples at specific quantiles.

        Parameters:
        -----------
        quantiles : float or array-like
            Quantile values in [0, 1].

        Returns:
        --------
        float or np.ndarray
            Samples at the specified quantiles.
        """
        inverse_cdf = self._get_inverse_cdf()
        return inverse_cdf(quantiles)


class InverseTransformSampler(BaseInverseTransformSampler):
    """
    Inverse transform sampler for distributions with known analytical inverse CDF.
    """

    def __init__(
        self,
        inverse_cdf: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        random_state: Optional[int] = None,
    ):
        """
        Parameters:
        -----------
        inverse_cdf : callable
            The analytical inverse CDF function.
        random_state : int, optional
            Random seed for reproducibility.
        """
        super().__init__(random_state)
        self.inverse_cdf_func = inverse_cdf

    def _get_inverse_cdf(self) -> Callable:
        return self.inverse_cdf_func


class NumericalInverseTransformSampler(BaseInverseTransformSampler):
    """
    Inverse transform sampler that computes inverse CDF numerically from a given CDF.
    """

    def __init__(
        self,
        cdf: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        x_range: Tuple[float, float],
        method: str = "interpolation",
        n_points: int = 1000,
        tolerance: float = 1e-8,
        random_state: Optional[int] = None,
    ):
        """
        Parameters:
        -----------
        cdf : callable
            The cumulative distribution function.
        x_range : tuple
            (min_x, max_x) - range of x values covering the distribution support.
        method : str
            'interpolation' (fast) or 'root_finding' (accurate).
        n_points : int
            Number of points for interpolation method.
        tolerance : float
            Tolerance for root finding method.
        random_state : int, optional
            Random seed for reproducibility.
        """
        super().__init__(random_state)
        self.cdf = cdf
        self.x_min, self.x_max = x_range
        self.method = method
        self.tolerance = tolerance
        self.n_points = n_points

        # Build numerical inverse
        self._build_inverse()

    def _build_inverse(self):
        """Build the numerical inverse CDF."""
        if self.method == "interpolation":
            self._build_interpolator()
        elif self.method == "root_finding":
            # No pre-computation needed for root finding
            pass
        else:
            raise ValueError("Method must be 'interpolation' or 'root_finding'")

    def _build_interpolator(self):
        """Build interpolation table for fast inverse lookup."""
        # Create x values spanning the range
        x_values = np.linspace(self.x_min, self.x_max, self.n_points)

        # Compute CDF values
        cdf_values = self.cdf(x_values)

        # Handle edge cases and ensure monotonicity
        cdf_values = np.clip(cdf_values, 0, 1)
        for i in range(1, len(cdf_values)):
            if cdf_values[i] < cdf_values[i - 1]:
                cdf_values[i] = cdf_values[i - 1]

        # Create interpolator
        self.interpolator = interp1d(
            cdf_values,
            x_values,
            kind="linear",
            bounds_error=False,
            fill_value=(self.x_min, self.x_max),
        )

    def _get_inverse_cdf(self) -> Callable:
        """Return the numerical inverse CDF function."""
        if self.method == "interpolation":
            return self._inverse_by_interpolation
        else:
            return self._inverse_by_root_finding

    def _inverse_by_interpolation(
        self, u: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Fast inverse using pre-computed interpolation table."""
        return self.interpolator(u)

    def _inverse_by_root_finding(
        self, u: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Accurate inverse using root finding for each u value."""
        u = np.asarray(u)
        is_scalar = u.ndim == 0
        u = np.atleast_1d(u)

        results = np.zeros_like(u)

        for i, u_val in enumerate(u):
            if u_val <= 0:
                results[i] = self.x_min
            elif u_val >= 1:
                results[i] = self.x_max
            else:

                def equation(x):
                    return self.cdf(x) - u_val

                try:
                    results[i] = brentq(
                        equation, self.x_min, self.x_max, xtol=self.tolerance
                    )
                except ValueError:
                    # Fallback to interpolation
                    if not hasattr(self, "interpolator"):
                        self._build_interpolator()
                    results[i] = self.interpolator(u_val)

        return results.item() if is_scalar else results


class AdaptiveInverseTransformSampler(NumericalInverseTransformSampler):
    """
    Adaptive inverse transform sampler that automatically refines grid resolution
    in regions where the CDF changes rapidly.
    """

    def __init__(
        self,
        cdf: Callable,
        x_range: Tuple[float, float],
        initial_points: int = 100,
        max_points: int = 10000,
        tolerance: float = 1e-6,
        random_state: Optional[int] = None,
    ):
        # Initialize with adaptive method
        super().__init__(
            cdf,
            x_range,
            method="interpolation",
            n_points=initial_points,
            random_state=random_state,
        )
        self.max_points = max_points
        self.adaptive_tolerance = tolerance

        # Override the standard interpolation with adaptive version
        self._build_adaptive_interpolator()

    def _build_adaptive_interpolator(self):
        """Build adaptive grid based on CDF curvature."""
        x_points = [self.x_min]
        cdf_points = [self.cdf(self.x_min)]

        # Initial uniform grid
        x_uniform = np.linspace(self.x_min, self.x_max, self.n_points)
        cdf_uniform = np.array([self.cdf(x) for x in x_uniform])

        # Add points where curvature is high
        for i in range(len(x_uniform) - 1):
            x_points.append(x_uniform[i])
            cdf_points.append(cdf_uniform[i])

            # Check curvature using second derivative approximation
            if i > 0 and i < len(x_uniform) - 1:
                d2_cdf = cdf_uniform[i + 1] - 2 * cdf_uniform[i] + cdf_uniform[i - 1]
                dx = x_uniform[i + 1] - x_uniform[i - 1]
                curvature = abs(d2_cdf) / (dx**2) if dx > 0 else 0

                # Add intermediate points if curvature is high
                if (
                    curvature > self.adaptive_tolerance
                    and len(x_points) < self.max_points
                ):
                    x_mid = (x_uniform[i] + x_uniform[i + 1]) / 2
                    x_points.append(x_mid)
                    cdf_points.append(self.cdf(x_mid))

        x_points.append(self.x_max)
        cdf_points.append(self.cdf(self.x_max))

        # Sort and create cubic spline interpolator
        sorted_indices = np.argsort(cdf_points)
        cdf_values = np.array(cdf_points)[sorted_indices]
        x_values = np.array(x_points)[sorted_indices]

        self.interpolator = CubicSpline(
            cdf_values, x_values, bc_type="natural", extrapolate=True
        )


class DiscreteInverseTransformSampler(BaseInverseTransformSampler):
    """
    Inverse transform sampler for discrete distributions.
    Implements Algorithm 2.3 from the textbook.
    """

    def __init__(
        self,
        values: np.ndarray,
        probabilities: np.ndarray,
        random_state: Optional[int] = None,
    ):
        """
        Parameters:
        -----------
        values : np.ndarray
            Array of discrete values that the random variable can take.
        probabilities : np.ndarray
            Array of probabilities corresponding to each value.
            Must sum to 1.0.
        random_state : int, optional
            Random seed for reproducibility.
        """
        super().__init__(random_state)

        # Validate inputs
        values = np.asarray(values)
        probabilities = np.asarray(probabilities)

        if len(values) != len(probabilities):
            raise ValueError("values and probabilities must have the same length")

        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("probabilities must sum to 1.0")

        if np.any(probabilities < 0):
            raise ValueError("probabilities must be non-negative")

        # Store sorted values and compute cumulative probabilities
        sorted_indices = np.argsort(values)
        self.values = values[sorted_indices]
        self.probabilities = probabilities[sorted_indices]
        self.cumulative_probs = np.cumsum(self.probabilities)

        # Ensure the last cumulative probability is exactly 1.0
        self.cumulative_probs[-1] = 1.0

    def _get_inverse_cdf(self) -> Callable:
        return self._discrete_inverse_cdf

    def _discrete_inverse_cdf(
        self, u: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Discrete inverse CDF using Algorithm 2.3.
        Find the smallest k such that F(x_k) >= U.
        """
        u = np.asarray(u)
        is_scalar = u.ndim == 0
        u = np.atleast_1d(u)

        # For each u, find the first cumulative probability >= u
        # This is equivalent to finding the smallest k such that F(x_k) >= u
        indices = np.searchsorted(self.cumulative_probs, u, side="right")

        # Clamp indices to valid range
        indices = np.clip(indices, 0, len(self.values) - 1)

        results = self.values[indices]

        return results.item() if is_scalar else results


class StratifiedInverseTransformSampler(BaseInverseTransformSampler):
    """
    Stratified inverse transform sampler for variance reduction.
    Works with both analytical and numerical inverse CDFs.
    """

    def __init__(
        self,
        base_sampler: BaseInverseTransformSampler,
        n_strata: int = 10,
        random_state: Optional[int] = None,
    ):
        """
        Parameters:
        -----------
        base_sampler : BaseInverseTransformSampler
            The base sampler (analytical or numerical).
        n_strata : int
            Number of strata to use.
        random_state : int, optional
            Random seed for reproducibility.
        """
        super().__init__(random_state)
        self.base_sampler = base_sampler
        self.n_strata = n_strata

    def _get_inverse_cdf(self) -> Callable:
        return self.base_sampler._get_inverse_cdf()

    def _generate_uniform(self, n: int) -> np.ndarray:
        """Generate stratified uniform samples."""
        samples = []
        samples_per_stratum = n // self.n_strata
        remaining_samples = n % self.n_strata

        for i in range(self.n_strata):
            # Define stratum boundaries
            lower = i / self.n_strata
            upper = (i + 1) / self.n_strata

            # Determine number of samples for this stratum
            stratum_samples = samples_per_stratum
            if i < remaining_samples:
                stratum_samples += 1

            # Generate stratified uniform samples
            u_base = np.random.uniform(0, 1, stratum_samples)
            u_stratified = lower + u_base * (upper - lower)
            samples.extend(u_stratified)

        return np.array(samples)


# Factory functions for easy creation
def create_sampler(
    inverse_cdf: Optional[Callable] = None,
    cdf: Optional[Callable] = None,
    x_range: Optional[Tuple[float, float]] = None,
    values: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    method: str = "analytical",
    **kwargs,
) -> BaseInverseTransformSampler:
    """
    Factory function to create appropriate sampler based on inputs.

    Parameters:
    -----------
    inverse_cdf : callable, optional
        Analytical inverse CDF function.
    cdf : callable, optional
        CDF function (for numerical methods).
    x_range : tuple, optional
        Range for numerical methods.
    values : np.ndarray, optional
        Discrete values for discrete method.
    probabilities : np.ndarray, optional
        Probabilities for discrete method.
    method : str
        'analytical', 'numerical', 'adaptive', 'discrete', 'stratified'.
    **kwargs : additional arguments passed to sampler constructor.

    Returns:
    --------
    BaseInverseTransformSampler
        Appropriate sampler instance.
    """
    if method == "analytical":
        if inverse_cdf is None:
            raise ValueError("inverse_cdf must be provided for analytical method")
        return InverseTransformSampler(inverse_cdf, **kwargs)

    elif method == "numerical":
        if cdf is None or x_range is None:
            raise ValueError("cdf and x_range must be provided for numerical method")
        return NumericalInverseTransformSampler(cdf, x_range, **kwargs)

    elif method == "adaptive":
        if cdf is None or x_range is None:
            raise ValueError("cdf and x_range must be provided for adaptive method")
        return AdaptiveInverseTransformSampler(cdf, x_range, **kwargs)

    elif method == "discrete":
        if values is None or probabilities is None:
            raise ValueError(
                "values and probabilities must be provided for discrete method"
            )
        return DiscreteInverseTransformSampler(values, probabilities, **kwargs)

    elif method == "stratified":
        if inverse_cdf is None:
            raise ValueError("inverse_cdf must be provided for analytical method")
        base_sampler = InverseTransformSampler(inverse_cdf, **kwargs)
        return StratifiedInverseTransformSampler(base_sampler, **kwargs)

    else:
        raise ValueError(f"Unknown method: {method}")
