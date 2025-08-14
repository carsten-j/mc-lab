import warnings

import numpy as np
from scipy.optimize import minimize_scalar


def psis(ratios, return_diagnostics=False):
    """
    Pareto Smoothed Importance Sampling (PSIS) procedure.

    Parameters
    ----------
    ratios : array-like
        Raw importance ratios r_s = p(θ_s)/g(θ_s)
    return_diagnostics : bool, optional
        If True, also return the estimated shape parameter k_hat

    Returns
    -------
    weights : ndarray
        PSIS-smoothed importance weights
    k_hat : float (only if return_diagnostics=True)
        Estimated shape parameter
    """
    ratios = np.asarray(ratios)
    S = len(ratios)

    # Sort ratios from lowest to highest
    sorted_indices = np.argsort(ratios)
    sorted_ratios = ratios[sorted_indices]

    # Step 1: Set M = floor(min(0.2*S, 3*sqrt(S)))
    M = int(min(0.2 * S, 3 * np.sqrt(S)))

    if M < 2:
        # Not enough tail samples for fitting
        warnings.warn("Sample size too small for PSIS")
        if return_diagnostics:
            return ratios, np.nan
        return ratios

    # Step 2: Initialize weights
    weights = sorted_ratios.copy()

    # Step 3: Set threshold u_hat
    u_hat = sorted_ratios[S - M]

    # Step 4: Estimate (k_hat, sigma_hat) for generalized Pareto distribution
    # Using the M largest importance ratios
    tail_ratios = sorted_ratios[S - M :]
    k_hat, sigma_hat = fit_generalized_pareto(tail_ratios - u_hat)

    # Apply regularization for small S (Appendix G)
    if S < 1000:
        k_hat = (M * k_hat + 10 * 0.5) / (M + 10)

    # Step 5: Replace the M largest weights with expected order statistics
    if k_hat < 1:  # Only smooth if k_hat is reasonable
        for z in range(1, M + 1):
            # Inverse CDF of generalized Pareto distribution
            p = (z - 0.5) / M
            if abs(k_hat) < 1e-10:  # k approximately 0 (exponential case)
                inv_cdf = -sigma_hat * np.log(1 - p)
            else:
                inv_cdf = u_hat + (sigma_hat / k_hat) * ((1 - p) ** (-k_hat) - 1)

            # Ensure we don't exceed the maximum observed ratio
            weights[S - M + z - 1] = min(inv_cdf, sorted_ratios[-1])

    # Unsort weights to match original order
    unsorted_weights = np.empty_like(weights)
    unsorted_weights[sorted_indices] = weights

    # Step 6: Check threshold and warn if necessary
    threshold = min(1 - 1 / np.log10(S) if S >= 10 else 0.5, 0.7)
    if k_hat > threshold:
        warnings.warn(
            f"Estimated shape parameter k_hat = {k_hat:.3f} exceeds threshold {threshold:.3f}. "
            "PSIS estimates may be unstable or have high bias."
        )

    if return_diagnostics:
        return unsorted_weights, k_hat
    return unsorted_weights


def fit_generalized_pareto(exceedances):
    """
    Fit generalized Pareto distribution using maximum likelihood estimation.
    Simplified version of Zhang & Stephens (2009) method.

    Parameters
    ----------
    exceedances : array-like
        Data exceeding the threshold (already threshold-subtracted)

    Returns
    -------
    k_hat : float
        Estimated shape parameter
    sigma_hat : float
        Estimated scale parameter
    """
    exceedances = np.asarray(exceedances)
    n = len(exceedances)

    if n < 2:
        return 0.5, np.mean(exceedances)

    # Method of moments for initial estimate
    m1 = np.mean(exceedances)
    m2 = np.mean(exceedances**2)

    if m2 > 2 * m1**2:
        # Initial estimates
        k_init = 0.5 * (m2 / m1**2 - 2)
        # sigma_init = m1 * (m2 / m1**2 - 1) / 2
    else:
        k_init = 0.1
        # sigma_init = m1

    # Constrain k to reasonable range
    k_init = np.clip(k_init, -0.5, 2.0)

    # Profile likelihood approach (simplified)
    def neg_log_likelihood(k):
        if abs(k) < 1e-10:
            # Exponential case
            sigma = np.mean(exceedances)
            if sigma <= 0:
                return np.inf
            return n * np.log(sigma) + np.sum(exceedances) / sigma
        else:
            # General case
            if k < 0 and np.any(exceedances > -1 / k):
                return np.inf

            # Profile out sigma
            if k > 0:
                sigma = np.mean(exceedances * (1 + k * exceedances) ** (1 / k - 1))
            else:
                temp = 1 + k * exceedances
                if np.any(temp <= 0):
                    return np.inf
                sigma = np.mean(exceedances / temp)

            if sigma <= 0:
                return np.inf

            # Compute negative log-likelihood
            temp = 1 + k * exceedances / sigma
            if np.any(temp <= 0):
                return np.inf

            return n * np.log(sigma) + (1 + 1 / k) * np.sum(np.log(temp))

    # Optimize
    try:
        result = minimize_scalar(
            neg_log_likelihood, bounds=(-0.5, 1.5), method="bounded"
        )
        k_hat = result.x
    except Exception:
        k_hat = k_init

    # Compute sigma given k
    if abs(k_hat) < 1e-10:
        sigma_hat = np.mean(exceedances)
    else:
        if k_hat > 0:
            sigma_hat = np.mean(
                exceedances * (1 + k_hat * exceedances) ** (1 / k_hat - 1)
            )
        else:
            sigma_hat = np.mean(exceedances / (1 + k_hat * exceedances))

    # Ensure sigma is positive
    sigma_hat = max(sigma_hat, 1e-10)

    return k_hat, sigma_hat


# Example usage and testing
if __name__ == "__main__":
    # Test with exponential example from the paper
    np.random.seed(42)

    # Target: Exponential(1), Proposal: Exponential(3)
    # This gives k = 1 - 1/3 = 2/3 ≈ 0.667
    n_samples = 1000
    proposal_samples = np.random.exponential(scale=1 / 3, size=n_samples)

    # Importance ratios
    # p(θ) = exp(-θ), g(θ) = 3*exp(-3θ)
    # r(θ) = exp(-θ) / (3*exp(-3θ)) = exp(2θ)/3
    ratios = np.exp(2 * proposal_samples) / 3

    # Apply PSIS
    weights, k_hat = psis(ratios, return_diagnostics=True)

    print(f"Sample size: {n_samples}")
    print("Theoretical k: 0.667")
    print(f"Estimated k_hat: {k_hat:.3f}")
    print(f"Mean raw ratio: {np.mean(ratios):.3f}")
    print(f"Mean PSIS weight: {np.mean(weights):.3f}")
    print(f"Max raw ratio: {np.max(ratios):.3f}")
    print(f"Max PSIS weight: {np.max(weights):.3f}")

    # Effective sample size
    normalized_weights = weights / np.sum(weights)
    ess = 1 / np.sum(normalized_weights**2) / n_samples
    print(f"Relative ESS: {ess:.3f}")
