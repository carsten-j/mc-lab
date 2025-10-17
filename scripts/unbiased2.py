from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CoupledChainResult:
    """Result from running coupled chains"""

    X_trajectory: list  # Trajectory of X chain
    Y_trajectory: list  # Trajectory of Y chain
    meeting_time: int  # Time when chains met (tau)
    X_final: np.ndarray  # Final state of X
    Y_final: np.ndarray  # Final state of Y


def maximal_coupling_normals(mu1, mu2, sigma):
    """
    Maximal coupling of two normal distributions
    N(mu1, sigma^2 I) and N(mu2, sigma^2 I)

    Returns: (X, Y) where X ~ N(mu1, sigma^2 I) and Y ~ N(mu2, sigma^2 I)
    """
    d = len(mu1)
    z = (mu1 - mu2) / sigma
    z_norm = np.linalg.norm(z)

    if z_norm < 1e-10:  # Distributions are essentially the same
        X = mu1 + sigma * np.random.normal(size=d)
        return X, X.copy()

    e = z / z_norm

    # Sample X from N(mu1, sigma^2 I)
    xi = np.random.normal(size=d)
    X = mu1 + sigma * xi

    # Compute acceptance probability for maximal coupling
    log_accept = -0.5 * z_norm * (z_norm + 2 * np.dot(xi, e))

    if np.log(np.random.uniform()) < log_accept:
        # Accept: Y = X + (mu2 - mu1)
        Y = X + (mu2 - mu1)
    else:
        # Reject: use reflection
        xi_reflected = xi - 2 * np.dot(xi, e) * e
        Y = mu2 + sigma * xi_reflected

    return X, Y


def coupled_mrth_improved(x, y, U, sigma):
    """
    Improved coupling of MRTH transition with Normal proposals

    Args:
        x: first chain state (numpy array)
        y: second chain state (numpy array)
        U: negative log-density function (callable)
        sigma: proposal standard deviation (scalar or array)

    Returns:
        tuple of (next_x, next_y)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Use maximal coupling for proposals
    xprop, yprop = maximal_coupling_normals(x, y, sigma)

    # Compute acceptance probabilities
    log_alpha_x = min(0, U(x) - U(xprop))
    log_alpha_y = min(0, U(y) - U(yprop))

    # Try to use common random numbers for acceptance
    u = np.random.uniform()

    # Accept/reject for X
    if np.log(u) < log_alpha_x:
        next_x = xprop
    else:
        next_x = x.copy()

    # Accept/reject for Y (using same u if possible)
    if np.log(u) < log_alpha_y:
        next_y = yprop
    else:
        next_y = y.copy()

    return next_x, next_y


def mrth(x, U, sigma):
    """
    Standard Metropolis-Rosenbluth-Teller-Hastings transition

    Args:
        x: current state (numpy array)
        U: negative log-density function (callable)
        sigma: proposal standard deviation (scalar or array)

    Returns:
        next state (numpy array)
    """
    x = np.asarray(x)
    # proposal = current location + Normal(0, sigma^2)
    xprop = x + sigma * np.random.normal(size=len(x))

    # log acceptance probability
    log_alpha = min(0, U(x) - U(xprop))

    # accept/reject
    if np.log(np.random.uniform()) < log_alpha:
        return xprop
    else:
        return x.copy()


def algorithm_1_coupled_chains(
    pi_0_sampler: Callable,
    P: Callable,
    P_bar: Callable,
    L: int,
    max_iterations: int,
    verbose: bool = False,
) -> CoupledChainResult:
    """
    Algorithm 1: Successful coupling of chains with lag L and length ℓ

    Args:
        pi_0_sampler: function that samples from initial distribution π_0
        P: standard transition kernel P(x, ·)
        P_bar: coupled transition kernel P̄((x,y), ·)
        L: lag parameter (L ≥ 1)
        max_iterations: maximum iterations (ℓ in the paper)
        verbose: print progress information

    Returns:
        CoupledChainResult with trajectories and meeting time
    """
    # Step 1: Sample (X_0, Y_0) from π̄_0
    X_0 = pi_0_sampler()
    Y_0 = pi_0_sampler()

    X_trajectory = [X_0]
    Y_trajectory = [Y_0]

    if verbose:
        print(f"Initial: X_0 = {X_0}, Y_0 = {Y_0}")
        print(f"Initial distance: {np.linalg.norm(X_0 - Y_0):.4f}")

    # Step 2: If L ≥ 1, for t = 1, ..., L, sample X_t from P(X_{t-1}, ·)
    for t in range(1, L + 1):
        X_t = P(X_trajectory[-1])
        X_trajectory.append(X_t)

    # For Y chain: run it forward for L-1 steps
    for t in range(1, L):
        Y_t = P(Y_trajectory[-1])
        Y_trajectory.append(Y_t)

    if verbose:
        print(
            f"After lag period: X_L = {X_trajectory[L]}, Y_{L - 1} = {Y_trajectory[L - 1]}"
        )

    # Step 3: For t ≥ L, sample (X_{t+1}, Y_{t-L+1}) from P̄((X_t, Y_{t-L}), ·)
    meeting_time = None
    t = L

    while t < max_iterations:
        # Get current states
        X_t = X_trajectory[t]
        Y_t_minus_L = Y_trajectory[t - L]

        # Sample coupled transition
        X_next, Y_next = P_bar(X_t, Y_t_minus_L)

        # Store new states
        X_trajectory.append(X_next)
        Y_trajectory.append(Y_next)

        # Check for meeting: X_{t+1} = Y_{t+1-L}
        # At time t+1, we have X_{t+1} and Y_{t+1-L}
        distance = np.linalg.norm(X_next - Y_next)

        if verbose and t % 100 == 0:
            print(
                f"t = {t}, distance between X_{t + 1} and Y_{t + 1 - L}: {distance:.6f}"
            )

        if distance < 1e-10 and meeting_time is None:
            meeting_time = t + 1  # τ = t + 1
            if verbose:
                print(f"Chains met at time {meeting_time}!")
                print(f"X_{meeting_time} = {X_next}")
                print(f"Y_{meeting_time - L} = {Y_next}")

        t += 1

        # After meeting, continue with perfect coupling
        if meeting_time is not None and t < max_iterations:
            # From the meeting time onwards, we can use perfect coupling
            # X_{s} = Y_{s-L} for all s ≥ τ
            while t < max_iterations:
                X_t = X_trajectory[t]
                # Perfect coupling: Y evolves the same as X but with lag L
                X_next = P(X_t)
                X_trajectory.append(X_next)
                # Y_{t+1-L} should equal X_{t+1} after meeting
                Y_trajectory.append(X_next.copy())
                t += 1
            break

    # If we never met, set meeting time to infinity
    if meeting_time is None:
        meeting_time = np.inf

    return CoupledChainResult(
        X_trajectory=X_trajectory,
        Y_trajectory=Y_trajectory,
        meeting_time=meeting_time,
        X_final=X_trajectory[-1],
        Y_final=Y_trajectory[-1],
    )


def run_unbiased_mcmc_example(L=5, sigma=1.5, max_iterations=2000, seed=None):
    """
    Example of using Algorithm 1 with coupled MRTH transitions

    Args:
        L: lag parameter
        sigma: proposal standard deviation
        max_iterations: maximum number of iterations
        seed: random seed
    """
    if seed is not None:
        np.random.seed(seed)

    # Define target distribution (standard normal in 2D)
    def U(x):
        """Negative log-density of a standard normal"""
        return 0.5 * np.sum(x**2)

    # Initial distribution sampler - start closer to target
    def pi_0_sampler():
        """Sample from initial distribution"""
        return np.random.normal(0, 2, size=2)  # Reduced from 5 to 2

    # Standard transition kernel
    def P(x):
        return mrth(x, U, sigma)

    # Coupled transition kernel
    def P_bar(x, y):
        return coupled_mrth_improved(x, y, U, sigma)

    # Run Algorithm 1
    print(f"Running with L={L}, sigma={sigma}, max_iterations={max_iterations}")

    result = algorithm_1_coupled_chains(
        pi_0_sampler=pi_0_sampler,
        P=P,
        P_bar=P_bar,
        L=L,
        max_iterations=max_iterations,
        verbose=True,
    )

    print("\n=== Results ===")
    print(f"Meeting time τ: {result.meeting_time}")
    print(f"Number of X iterations: {len(result.X_trajectory)}")
    print(f"Number of Y iterations: {len(result.Y_trajectory)}")
    print(f"Final X: {result.X_final}")
    print(f"Final Y: {result.Y_final}")

    if result.meeting_time < np.inf:
        print(f"Chains successfully met at time {result.meeting_time}")
    else:
        print("Chains did not meet within max_iterations")
        print("Try: smaller L, larger sigma, or more iterations")

    return result


# Run multiple experiments to test different parameters
if __name__ == "__main__":
    print("Experiment 1: Small lag, moderate proposal variance")
    result1 = run_unbiased_mcmc_example(L=1, sigma=5, max_iterations=20000, seed=42)

    # print("\n" + "=" * 50 + "\n")
    # print("Experiment 2: Moderate lag, larger proposal variance")
    # result2 = run_unbiased_mcmc_example(L=5, sigma=2.0, max_iterations=1500, seed=43)

    # print("\n" + "=" * 50 + "\n")
    # print("Experiment 3: Larger lag, optimal proposal variance")
    # result3 = run_unbiased_mcmc_example(L=10, sigma=2.4, max_iterations=2000, seed=44)

    # Try visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        # Plot the first successful result
        for result in [result1]:  # , result2, result3]:
            if result.meeting_time < np.inf:
                X_traj = np.array(result.X_trajectory)
                Y_traj = np.array(result.Y_trajectory)

                plt.figure(figsize=(10, 5))

                # Plot both chains
                plt.subplot(1, 2, 1)
                plt.plot(
                    X_traj[:, 0],
                    X_traj[:, 1],
                    "b-",
                    alpha=0.5,
                    label="X chain",
                    linewidth=0.5,
                )
                plt.plot(
                    Y_traj[:, 0],
                    Y_traj[:, 1],
                    "r-",
                    alpha=0.5,
                    label="Y chain",
                    linewidth=0.5,
                )
                plt.scatter(X_traj[0, 0], X_traj[0, 1], c="blue", s=100, marker="o")
                plt.scatter(Y_traj[0, 0], Y_traj[0, 1], c="red", s=100, marker="o")

                # Mark meeting point if it exists
                tau = int(result.meeting_time)
                if tau < len(X_traj):
                    plt.scatter(
                        X_traj[tau, 0],
                        X_traj[tau, 1],
                        c="green",
                        s=200,
                        marker="*",
                        label=f"Meeting at τ={tau}",
                        zorder=5,
                    )

                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.title(f"Coupled Chains (L={tau}, meeting time={tau})")
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Plot distance over time
                plt.subplot(1, 2, 2)
                min_len = min(len(X_traj), len(Y_traj))
                distances = [
                    np.linalg.norm(X_traj[i] - Y_traj[min(i, len(Y_traj) - 1)])
                    for i in range(min_len)
                ]
                plt.plot(distances, "g-", linewidth=1)
                plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
                if tau < len(distances):
                    plt.axvline(
                        x=tau,
                        color="b",
                        linestyle="--",
                        alpha=0.5,
                        label=f"Meeting time τ={tau}",
                    )
                plt.xlabel("Iteration")
                plt.ylabel("Distance between chains")
                plt.title("Chain coupling distance")
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()
                break

    except ImportError:
        print("\nMatplotlib not available for visualization")
