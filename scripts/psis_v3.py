import warnings

import numpy as np
from scipy import stats


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
    exceedances = tail_ratios - u_hat

    # Fit generalized Pareto distribution using scipy
    # scipy.stats.genpareto uses shape parameter c (our k), loc=0, and scale (our sigma)
    try:
        # Method of moments for initial guess (helps convergence)
        m1 = np.mean(exceedances)
        m2 = np.mean(exceedances**2)

        if m2 > 2 * m1**2:
            k_init = 0.5 * (m2 / m1**2 - 2)
            sigma_init = m1 * (m2 / m1**2 - 1) / 2
        else:
            k_init = 0.1
            sigma_init = m1

        # Fit using MLE with initial guess
        # Note: scipy's shape parameter c corresponds to our k
        params = stats.genpareto.fit(
            exceedances, floc=0, fc=k_init, fscale=sigma_init, method="MLE"
        )
        k_hat, loc, sigma_hat = params

        # Ensure loc is 0 (we already subtracted the threshold)
        if abs(loc) > 1e-10:
            warnings.warn("Location parameter should be 0 for exceedances")

    except Exception:
        # Fallback to method of moments
        m1 = np.mean(exceedances)
        m2 = np.var(exceedances)
        if m2 > 0:
            k_hat = 0.5 * (m1**2 / m2 - 1)
            sigma_hat = 0.5 * m1 * (m1**2 / m2 + 1)
        else:
            k_hat = 0.5
            sigma_hat = m1

    # Constrain k_hat to reasonable range
    k_hat = np.clip(k_hat, -0.5, 2.0)
    sigma_hat = max(sigma_hat, 1e-10)

    # Apply regularization for small S (Appendix G)
    if S < 1000:
        k_hat = (M * k_hat + 10 * 0.5) / (M + 10)

    # Step 5: Replace the M largest weights with expected order statistics
    if k_hat < 1:  # Only smooth if k_hat is reasonable
        for z in range(1, M + 1):
            # Inverse CDF of generalized Pareto distribution
            p = (z - 0.5) / M

            # Use scipy's ppf (percent point function = inverse CDF)
            # Add back the threshold u_hat
            inv_cdf = u_hat + stats.genpareto.ppf(p, c=k_hat, loc=0, scale=sigma_hat)

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


def psis_with_custom_fit(ratios, return_diagnostics=False, method="scipy"):
    """
    PSIS with option to use custom fitting method.

    Parameters
    ----------
    ratios : array-like
        Raw importance ratios
    return_diagnostics : bool
        If True, return k_hat
    method : str
        'scipy' for scipy.stats.genpareto.fit
        'zhang-stephens' for custom implementation (more robust for small samples)
    """
    if method == "scipy":
        return psis(ratios, return_diagnostics)
    else:
        # Could implement Zhang-Stephens method here if needed
        raise NotImplementedError(
            "Zhang-Stephens method not implemented in this version"
        )


# Example usage and testing
if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: Exponential example from the paper
    print("Test 1: Exponential example")
    print("-" * 40)
    n_samples = 1000
    proposal_samples = np.random.exponential(scale=1 / 3, size=n_samples)

    # Importance ratios
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

    # Test 2: Compare with different sample sizes
    print("\nTest 2: Effect of sample size")
    print("-" * 40)
    for n_samples in [100, 500, 1000, 5000]:
        proposal_samples = np.random.exponential(scale=1 / 3, size=n_samples)
        ratios = np.exp(2 * proposal_samples) / 3
        weights, k_hat = psis(ratios, return_diagnostics=True)

        normalized_weights = weights / np.sum(weights)
        ess = 1 / np.sum(normalized_weights**2) / n_samples

        print(f"N={n_samples:5d}: k_hat={k_hat:.3f}, Relative ESS={ess:.3f}")

    # Test 3: High k case (should trigger warning)
    print("\nTest 3: High k case (should trigger warning)")
    print("-" * 40)
    # Use proposal with much lighter tail
    proposal_samples = np.random.exponential(scale=1 / 10, size=1000)
    ratios = np.exp(9 * proposal_samples) / 10  # k ≈ 0.9

    weights, k_hat = psis(ratios, return_diagnostics=True)
    print(f"Estimated k_hat: {k_hat:.3f}")
