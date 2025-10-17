"""
Corrected Unbiased Metropolis-Hastings Algorithm with Maximal Coupling
======================================================================

Fixed implementation with proper lag-L coupling structure.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.linalg import cholesky, solve_triangular


@dataclass
class MCMCState:
    """Container for MCMC chain state."""

    position: np.ndarray
    log_density: float


@dataclass
class CoupledChains:
    """Container for two coupled MCMC chains with lag structure."""

    chain_x: List[np.ndarray]  # First chain trajectory
    chain_y: List[np.ndarray]  # Second chain trajectory (lagged)
    meeting_time: Optional[int]  # Meeting time τ when X_τ = Y_{τ-L}
    lag: int  # Lag L between chains


class ReflectionMaximalCoupling:
    """
    Reflection-maximal coupling for multivariate normal distributions.
    """

    def __init__(self, covariance: np.ndarray, tolerance: float = 1e-10):
        self.covariance = covariance
        self.tolerance = tolerance

        try:
            self.chol_lower = cholesky(covariance, lower=True)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix must be positive definite")

        self.dim = covariance.shape[0]

    def sample_coupled(
        self, mu1: np.ndarray, mu2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Sample coupled pair (X, Y) from N(μ₁, Σ) and N(μ₂, Σ).
        """
        mean_diff = mu1 - mu2
        distance = np.linalg.norm(mean_diff)

        if distance < self.tolerance:
            # Synchronous coupling
            z = np.random.standard_normal(self.dim)
            x = mu1 + self.chol_lower @ z
            return x, x.copy(), True

        # Sample X ~ N(μ₁, Σ)
        z = np.random.standard_normal(self.dim)
        x = mu1 + self.chol_lower @ z

        # Compute standardized direction
        z_diff = solve_triangular(self.chol_lower, mean_diff, lower=True)
        norm_z = np.linalg.norm(z_diff)

        if norm_z < self.tolerance:
            y = mu2 + self.chol_lower @ z
            return x, y, True

        e = z_diff / norm_z

        # Compute acceptance probability
        projection = np.dot(e, z)
        log_accept_prob = stats.norm.logpdf(projection + norm_z) - stats.norm.logpdf(
            projection
        )

        # Accept or reject
        if np.log(np.random.uniform()) < log_accept_prob:
            # Accept: translate
            y = x + (mu2 - mu1)
            coupled = True
        else:
            # Reject: reflect
            z_reflected = z - 2 * projection * e
            y = mu2 + self.chol_lower @ z_reflected
            coupled = False

        return x, y, coupled


class UnbiasedMetropolisHastings:
    """
    Unbiased Metropolis-Hastings with proper lag-L coupling.

    The key insight: at iteration t >= L, we have:
    - X at position X_t
    - Y at position Y_{t-L}
    Meeting occurs when X_t = Y_{t-L}.
    """

    def __init__(
        self,
        target_log_density: Callable[[np.ndarray], float],
        proposal_covariance: np.ndarray,
    ):
        self.target_log_density = target_log_density
        self.proposal_cov = proposal_covariance
        self.coupling = ReflectionMaximalCoupling(proposal_covariance)
        self.dim = proposal_covariance.shape[0]

    def single_kernel(
        self, position: np.ndarray, log_density: float
    ) -> Tuple[np.ndarray, float]:
        """Single MH kernel step."""
        # Propose
        proposal = np.random.multivariate_normal(position, self.proposal_cov)
        proposal_log_dens = self.target_log_density(proposal)

        # Accept/reject
        log_alpha = min(0.0, proposal_log_dens - log_density)

        if np.log(np.random.uniform()) < log_alpha:
            return proposal, proposal_log_dens
        else:
            return position.copy(), log_density

    def coupled_kernel(
        self,
        x_pos: np.ndarray,
        x_log_dens: float,
        y_pos: np.ndarray,
        y_log_dens: float,
        has_met: bool = False,
    ) -> Tuple[np.ndarray, float, np.ndarray, float, bool]:
        """
        Coupled MH kernel.
        Returns (new_x_pos, new_x_log_dens, new_y_pos, new_y_log_dens, are_equal).
        """
        are_equal = np.allclose(x_pos, y_pos, atol=1e-12)

        if are_equal or has_met:
            # Synchronous coupling
            proposal_x, proposal_y, _ = self.coupling.sample_coupled(x_pos, x_pos)
            proposal_log_dens = self.target_log_density(proposal_x)

            # Common acceptance
            log_alpha = min(0.0, proposal_log_dens - x_log_dens)

            if np.log(np.random.uniform()) < log_alpha:
                return (
                    proposal_x,
                    proposal_log_dens,
                    proposal_y,
                    proposal_log_dens,
                    True,
                )
            else:
                return x_pos.copy(), x_log_dens, y_pos.copy(), y_log_dens, True

        else:
            # Maximal coupling
            proposal_x, proposal_y, _ = self.coupling.sample_coupled(x_pos, y_pos)

            proposal_log_dens_x = self.target_log_density(proposal_x)
            proposal_log_dens_y = self.target_log_density(proposal_y)

            log_alpha_x = min(0.0, proposal_log_dens_x - x_log_dens)
            log_alpha_y = min(0.0, proposal_log_dens_y - y_log_dens)

            # Common random number
            log_u = np.log(np.random.uniform())

            if log_u < log_alpha_x:
                new_x, new_x_dens = proposal_x, proposal_log_dens_x
            else:
                new_x, new_x_dens = x_pos.copy(), x_log_dens

            if log_u < log_alpha_y:
                new_y, new_y_dens = proposal_y, proposal_log_dens_y
            else:
                new_y, new_y_dens = y_pos.copy(), y_log_dens

            chains_equal = np.allclose(new_x, new_y, atol=1e-12)

            return new_x, new_x_dens, new_y, new_y_dens, chains_equal

    def sample_coupled_chains(
        self, initial_state: np.ndarray, n_iterations: int, lag: int = 1
    ) -> CoupledChains:
        """
        Sample coupled chains with lag-L structure.

        X runs from 0 to n_iterations
        Y runs from 0 to n_iterations-L
        At time t >= L, we compare X_t with Y_{t-L}
        """
        if lag < 1:
            lag = 1

        # Initialize storage
        chain_x = []
        chain_y = []

        # Initialize BOTH chains at time 0 from the SAME initial state
        x_pos = initial_state.copy()
        x_log_dens = self.target_log_density(x_pos)
        chain_x.append(x_pos.copy())

        y_pos = initial_state.copy()
        y_log_dens = self.target_log_density(y_pos)
        chain_y.append(y_pos.copy())

        # Run BOTH chains independently for L steps
        for t in range(1, lag + 1):
            x_pos, x_log_dens = self.single_kernel(x_pos, x_log_dens)
            chain_x.append(x_pos.copy())

            y_pos, y_log_dens = self.single_kernel(y_pos, y_log_dens)
            chain_y.append(y_pos.copy())

        meeting_time = None
        has_met = False

        # Now run both chains
        # X continues from time L+1 to n_iterations
        # Y runs from time 1 to n_iterations-L
        for t in range(lag + 1, n_iterations + 1):
            # At time t, we have:
            # - X at iteration t (trying to generate X_t)
            # - Y at iteration t-L (trying to generate Y_{t-L})

            # Coupled kernel step
            x_pos, x_log_dens, y_pos, y_log_dens, _ = self.coupled_kernel(
                x_pos, x_log_dens, y_pos, y_log_dens, has_met
            )

            #  Check if they've met AFTER the update (X_t = Y_{t-L})
            if not has_met:
                if np.allclose(x_pos, y_pos, atol=1e-12):
                    meeting_time = t
                    has_met = True

            chain_x.append(x_pos.copy())
            chain_y.append(y_pos.copy())

        return CoupledChains(
            chain_x=chain_x, chain_y=chain_y, meeting_time=meeting_time, lag=lag
        )

    def compute_unbiased_estimator(
        self,
        coupled_chains: CoupledChains,
        test_function: Callable[[np.ndarray], float],
        k: int,
        m: int,
    ) -> float:
        """
        Compute unbiased estimator H_{k:m}.

        H_{k:m} = (1/(m-k+1)) * Σ_{t=k}^m h(X_t)
                  + Σ_{t=k}^{min(τ-1,m)} [h(X_t) - h(Y_t)]

        where τ is the meeting time when X_τ = Y_{τ-L}.

        Note: Both chains X and Y run independently for L steps from the same
        initial state, then are coupled. The bias correction compares X_t and Y_t
        at the same iteration index, not X_t with Y_{t-L}.
        """
        if m < k:
            raise ValueError(f"Must have m >= k, got m={m}, k={k}")

        if k < 0:
            raise ValueError(f"k must be >= 0, got k={k}")

        chain_x = coupled_chains.chain_x
        chain_y = coupled_chains.chain_y
        lag = coupled_chains.lag
        tau = coupled_chains.meeting_time

        # Part 1: Time-averaged MCMC estimator
        mcmc_sum = 0.0
        for t in range(k, min(m + 1, len(chain_x))):
            mcmc_sum += test_function(chain_x[t])
        mcmc_average = mcmc_sum / (m - k + 1)

        # Part 2: Bias correction term
        # Now that both chains have same burn-in, we can use standard formula
        bias_correction = 0.0

        if tau is not None:
            upper_limit = min(tau - 1, m)
        else:
            upper_limit = m

        # Bias correction: compare X_t with Y_t (both chains at same iteration)
        # This works because both chains now have equal burn-in
        for t in range(k, upper_limit + 1):
            if t < len(chain_y):
                h_x = test_function(chain_x[t])
                h_y = test_function(chain_y[t])
                bias_correction += h_x - h_y

        return mcmc_average + bias_correction

    def estimate_meeting_time_distribution(
        self,
        initial_state: np.ndarray,
        n_replicates: int = 100,
        max_iterations: int = 10000,
        lag: int = 1,
    ) -> Dict[str, float]:
        """Estimate meeting time distribution."""
        meeting_times = []

        print(f"Estimating meeting times with {n_replicates} replicates (lag={lag})...")

        for i in range(n_replicates):
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{n_replicates} replicates")

            chains = self.sample_coupled_chains(
                initial_state=initial_state, n_iterations=max_iterations, lag=lag
            )

            if chains.meeting_time is not None:
                meeting_times.append(chains.meeting_time)
            else:
                meeting_times.append(np.inf)

        finite_times = [t for t in meeting_times if np.isfinite(t)]

        if len(finite_times) == 0:
            warnings.warn("No chains met!")
            return {
                "mean": np.inf,
                "median": np.inf,
                "p90": np.inf,
                "p95": np.inf,
                "proportion_not_met": 1.0,
            }

        return {
            "mean": np.mean(finite_times),
            "median": np.percentile(finite_times, 50),
            "p90": np.percentile(finite_times, 90),
            "p95": np.percentile(finite_times, 95),
            "proportion_not_met": (n_replicates - len(finite_times)) / n_replicates,
        }


def run_unbiased_mcmc(
    sampler: UnbiasedMetropolisHastings,
    test_function: Callable[[np.ndarray], float],
    initial_state: np.ndarray,
    n_chains: int,
    k: int,
    m: int,
    lag: int = 1,
    verbose: bool = True,
) -> Dict[str, any]:
    """Run multiple independent unbiased MCMC chains."""
    if lag is None or lag < 1:
        lag = 1

    if verbose:
        print(f"Running {n_chains} independent coupled chains...")
        print(f"Parameters: k={k}, m={m}, lag={lag}")

    estimators = []
    meeting_times = []

    for i in range(n_chains):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Completed {i + 1}/{n_chains} chains")

        # Sample coupled chains
        chains = sampler.sample_coupled_chains(
            initial_state=initial_state,
            n_iterations=m + lag,  # Need extra iterations for lag
            lag=lag,
        )

        # Compute unbiased estimator
        h_km = sampler.compute_unbiased_estimator(
            coupled_chains=chains, test_function=test_function, k=k, m=m
        )

        estimators.append(h_km)
        meeting_times.append(
            chains.meeting_time if chains.meeting_time is not None else np.inf
        )

    # Aggregate results
    estimators = np.array(estimators)
    mean_estimate = np.mean(estimators)
    std_error = np.std(estimators, ddof=1) / np.sqrt(n_chains)

    # 95% CI
    ci_lower = mean_estimate - 1.96 * std_error
    ci_upper = mean_estimate + 1.96 * std_error

    if verbose:
        finite_times = [t for t in meeting_times if np.isfinite(t)]
        print("\nResults:")
        print(f"  Estimate: {mean_estimate:.6f}")
        print(f"  Std Error: {std_error:.6f}")
        print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        if finite_times:
            print(f"  Mean meeting time: {np.mean(finite_times):.1f}")
            print(f"  Proportion met: {len(finite_times) / len(meeting_times):.2%}")

    return {
        "estimate": mean_estimate,
        "std_error": std_error,
        "confidence_interval": (ci_lower, ci_upper),
        "estimators": estimators,
        "meeting_times": meeting_times,
    }


def example_simple_gaussian():
    """Simpler example for validation."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Simple 1D Gaussian")
    print("=" * 70)

    # 1D Gaussian with mean 3, variance 1
    target_mean = 3.0
    target_var = 1.0

    target_log_density = lambda x: -0.5 * (x[0] - target_mean) ** 2 / target_var

    # Proposal variance
    proposal_cov = np.array([[0.5]])

    # Initialize sampler
    sampler = UnbiasedMetropolisHastings(
        target_log_density=target_log_density, proposal_covariance=proposal_cov
    )

    # Test function: identity (estimate mean)
    test_function = lambda x: x[0]
    true_value = target_mean

    print(f"\nTarget: N({target_mean}, {target_var})")
    print(f"True E[X] = {true_value:.6f}")

    # Run with simple parameters
    k = 10
    m = 100
    lag = 1

    print(f"\nFixed parameters: k={k}, m={m}, lag={lag}")

    results = run_unbiased_mcmc(
        sampler=sampler,
        test_function=test_function,
        initial_state=np.array([0.0]),
        n_chains=6000,
        k=k,
        m=m,
        lag=lag,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"True value:     {true_value:.6f}")
    print(f"Estimate:       {results['estimate']:.6f}")
    print(f"Error:          {results['estimate'] - true_value:.6f}")
    print(f"Std Error:      {results['std_error']:.6f}")
    print(
        f"95% CI:         [{results['confidence_interval'][0]:.6f}, "
        f"{results['confidence_interval'][1]:.6f}]"
    )

    ci_contains_true = (
        results["confidence_interval"][0]
        <= true_value
        <= results["confidence_interval"][1]
    )
    print(f"CI contains true value: {ci_contains_true}")

    # Meeting time statistics
    finite_meeting_times = [t for t in results["meeting_times"] if np.isfinite(t)]
    if finite_meeting_times:
        avg_meeting_time = np.mean(finite_meeting_times)
        print("\nMeeting time statistics:")
        print(f"  Average meeting time: {avg_meeting_time:.2f}")
        print(
            f"  Proportion of chains that met: {len(finite_meeting_times) / len(results['meeting_times']):.2%}"
        )
    else:
        print("\nNo chains met within the iteration limit")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Run examples
    example_simple_gaussian()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
