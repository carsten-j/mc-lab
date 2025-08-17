import numpy as np
from scipy.stats import genpareto


def pareto_smooth_is(weights, tail_fraction=0.2):
    """
    Pareto Smoothed Importance Sampling (PSIS).

    Given an array of importance weights, returns smoothed weights and the Pareto shape parameter k.
    The largest weights are smoothed by fitting a generalized Pareto distribution (GPD) to the upper tail
    of the weight distribution and replacing those extreme weights with expected order statistics from the GPD ([arxiv.org](https://arxiv.org/html/1507.02646v9#:~:text=We%20propose%20a%20new%20method,tails%20of%20the%20importance%20distribution)).

    Parameters:
        weights (array_like): 1D array of raw importance weights (positive values).
        tail_fraction (float): Fraction of samples to consider as the "upper tail".
                                For example, 0.2 means the top 20% of weights will be modeled by GPD.

    Returns:
        smoothed_weights (ndarray): Array of the same shape as input, with largest weights replaced by smoothed values.
        k_hat (float): Estimated shape parameter of the fitted Pareto distribution (diagnostic).
    """
    weights = np.asarray(weights, dtype=float)
    N = weights.size
    if N == 0:
        return np.array([]), np.nan  # handle empty input

    # Step 1: Sort weights in ascending order for convenience
    sorted_weights = np.sort(weights)

    # Step 2: Determine tail threshold index based on tail_fraction
    tail_count = int(
        max(3, np.floor(tail_fraction * N))
    )  # at least 3 points in tail for fit
    if tail_count >= N:
        tail_count = N - 1  # ensure at least one weight remains non-tail if possible

    # Identify the cutoff index for the tail and the threshold value
    threshold_index = N - tail_count
    threshold = sorted_weights[threshold_index]  # smallest weight in the tail segment

    # Extract the upper tail weights (those to be smoothed)
    tail_weights = sorted_weights[threshold_index:]
    excesses = tail_weights - threshold  # convert to "excess over threshold"

    # Step 3: Fit a Generalized Pareto Distribution (GPD) to the excesses above the threshold
    # We fix loc=0 for the excess distribution. genpareto.fit returns (shape, loc, scale).
    xi, _, sigma = genpareto.fit(excesses, floc=0)
    k_hat = xi  # Pareto shape parameter (tail index)

    # Step 4: Replace the tail weights with expected order statistics from the fitted GPD
    M = tail_weights.size  # number of tail points being smoothed
    # We'll generate expected values for the order statistics of M samples from the GPD.
    smoothed_excesses = np.empty(M)
    for i in range(1, M + 1):
        # Use the quantile function of the GPD for probability point p = i/(M+1)
        p = i / (M + 1.0)
        if k_hat != 0:
            # inverse CDF for GPD: z(p) = sigma/k * [ (1-p)^(-k) - 1 ]
            smoothed_excesses[i - 1] = (sigma / k_hat) * ((1 - p) ** (-k_hat) - 1.0)
        else:
            # limit case k -> 0 (GPD becomes exponential): z(p) = sigma * log(1/(1-p))
            smoothed_excesses[i - 1] = sigma * np.log(1.0 / (1 - p))
    smoothed_tail = threshold + smoothed_excesses  # add back the threshold

    # The smoothed_tail array is sorted in ascending order (it corresponds to sorted_weights[threshold_index:]).
    # Now reconstruct the full array of smoothed weights: lower part remains the same, upper part replaced.
    smoothed_weights = np.empty_like(sorted_weights)
    smoothed_weights[:threshold_index] = sorted_weights[
        :threshold_index
    ]  # weights below threshold unchanged
    smoothed_weights[threshold_index:] = smoothed_tail  # tail weights smoothed

    return smoothed_weights, k_hat


# Example usage:
# Suppose we have importance weights, for demonstration we'll simulate some.
# Let's create a weight distribution with a heavy tail:
# Most weights around 1, but a few extremely large values (to simulate outliers).
np.random.seed(0)
base_weights = np.random.exponential(scale=1.0, size=1000)  # 1000 weights ~ Exp(1)
outliers = np.random.uniform(
    low=50, high=100, size=5
)  # 5 large weights between 50 and 100
weights = np.concatenate([base_weights, outliers])

# Apply PSIS smoothing
smoothed_w, k_hat_est = pareto_smooth_is(weights, tail_fraction=0.2)
print(f"Estimated Pareto shape k = {k_hat_est:.3f}")
print(f"Original max weight = {weights.max():.2f}")
print(f"Smoothed max weight = {smoothed_w.max():.2f}")
print(
    f"Number of weights = {len(weights)}, Effective tail count = {len(weights) - np.searchsorted(np.sort(weights), smoothed_w[len(weights) - len(outliers)])}"
)
