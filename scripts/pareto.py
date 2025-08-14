import warnings

import numpy as np
from scipy.optimize import minimize


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
        return 0.5, np.mean(exceedances) if len(exceedances) > 0 else 1.0

    # Method of moments for initial estimates
    m1 = np.mean(exceedances)
    m2 = np.mean(exceedances**2)

    if m2 > 2 * m1**2:
        # Initial estimates based on method of moments
        k_init = 0.5 * (m2 / m1**2 - 2)
        sigma_init = m1 * (m2 / m1**2 - 1) / 2
    else:
        k_init = 0.1
        sigma_init = m1

    # Constrain initial values to reasonable ranges
    k_init = np.clip(k_init, -0.5, 2.0)
    sigma_init = max(sigma_init, 1e-10)

    def neg_log_likelihood(params):
        """Negative log-likelihood for generalized Pareto distribution."""
        k, sigma = params

        if sigma <= 0:
            return np.inf

        if abs(k) < 1e-10:
            # Exponential case (k → 0)
            return n * np.log(sigma) + np.sum(exceedances) / sigma
        else:
            # General case
            temp = 1 + k * exceedances / sigma

            # Check domain constraints
            if np.any(temp <= 0):
                return np.inf

            return n * np.log(sigma) + (1 + 1 / k) * np.sum(np.log(temp))

    # Joint optimization over both k and sigma
    try:
        result = minimize(
            neg_log_likelihood,
            x0=[k_init, sigma_init],
            bounds=[(-0.5, 1.5), (1e-10, None)],
            method="L-BFGS-B",
        )

        if result.success:
            k_hat, sigma_hat = result.x
        else:
            # Fall back to initial estimates if optimization fails
            k_hat, sigma_hat = k_init, sigma_init

    except Exception as e:
        warnings.warn(f"Optimization failed: {e}. Using initial estimates.")
        k_hat, sigma_hat = k_init, sigma_init

    return k_hat, sigma_hat


def fit_generalized_pareto_profile(exceedances):
    """
    Alternative implementation using profile likelihood (optimizing only over k).
    This is what the original code intended but didn't use sigma_init properly.

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
        return 0.5, np.mean(exceedances) if len(exceedances) > 0 else 1.0

    # Method of moments for initial estimate of k only
    m1 = np.mean(exceedances)
    m2 = np.mean(exceedances**2)

    if m2 > 2 * m1**2:
        k_init = 0.5 * (m2 / m1**2 - 2)
    else:
        k_init = 0.1

    k_init = np.clip(k_init, -0.5, 2.0)

    def profile_neg_log_likelihood(k):
        """Profile negative log-likelihood, with sigma profiled out."""
        if abs(k) < 1e-10:
            # Exponential case
            sigma = np.mean(exceedances)
            if sigma <= 0:
                return np.inf
            return n * np.log(sigma) + n  # Since sum(x)/mean(x) = n
        else:
            # Compute profile MLE for sigma given k
            if k > 0:
                temp = 1 + k * exceedances
                if np.any(temp <= 0):
                    return np.inf
                # Profile MLE for sigma
                sigma = (1 + 1 / k) * np.mean(exceedances * temp ** (-1 / k))
            else:
                temp = 1 + k * exceedances
                if np.any(temp <= 0):
                    return np.inf
                # Profile MLE for sigma
                sigma = np.mean(exceedances * (1 + k * exceedances) ** (-1))

            if sigma <= 0:
                return np.inf

            # Compute negative log-likelihood at profile sigma
            temp = 1 + k * exceedances / sigma
            if np.any(temp <= 0):
                return np.inf

            return n * np.log(sigma) + (1 + 1 / k) * np.sum(np.log(temp))

    # Optimize only over k
    from scipy.optimize import minimize_scalar

    try:
        result = minimize_scalar(
            profile_neg_log_likelihood, bounds=(-0.5, 1.5), method="bounded"
        )
        k_hat = result.x
    except Exception:
        k_hat = k_init

    # Compute sigma given optimal k
    if abs(k_hat) < 1e-10:
        sigma_hat = np.mean(exceedances)
    else:
        if k_hat > 0:
            temp = 1 + k_hat * exceedances
            sigma_hat = (1 + 1 / k_hat) * np.mean(exceedances * temp ** (-1 / k_hat))
        else:
            sigma_hat = np.mean(exceedances * (1 + k_hat * exceedances) ** (-1))

    sigma_hat = max(sigma_hat, 1e-10)

    return k_hat, sigma_hat


# Test the implementations
if __name__ == "__main__":
    np.random.seed(42)

    # Generate some test data from a known GPD
    true_k = 0.3
    true_sigma = 2.0
    n_samples = 100

    # Generate GPD samples using inverse transform
    u = np.random.uniform(0, 1, n_samples)
    if abs(true_k) < 1e-10:
        samples = -true_sigma * np.log(1 - u)
    else:
        samples = true_sigma * ((1 - u) ** (-true_k) - 1) / true_k

    print("Comparison of fitting methods:")
    print(f"True parameters: k={true_k:.3f}, sigma={true_sigma:.3f}")
    print("-" * 50)

    # Test joint optimization
    k_joint, sigma_joint = fit_generalized_pareto(samples)
    print(f"Joint optimization: k={k_joint:.3f}, sigma={sigma_joint:.3f}")

    # Test profile likelihood
    k_profile, sigma_profile = fit_generalized_pareto_profile(samples)
    print(f"Profile likelihood: k={k_profile:.3f}, sigma={sigma_profile:.3f}")

    # Test scipy
    from scipy import stats

    params = stats.genpareto.fit(samples, floc=0)
    k_scipy, _, sigma_scipy = params
    print(f"Scipy genpareto:    k={k_scipy:.3f}, sigma={sigma_scipy:.3f}")
