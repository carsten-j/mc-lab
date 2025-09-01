import numpy as np
import pytest
import scipy.stats as stats

from mc_lab.inverse_transform import (
    create_sampler,
)


def test_create_sampler():
    def exp_inverse_cdf(u, rate=2.0):
        return -np.log(1 - u) / rate

    # Create analytical sampler
    exp_sampler = create_sampler(
        inverse_cdf=lambda u: exp_inverse_cdf(u, rate=2.0),
        method="analytical",
        random_state=42,
    )

    samples = exp_sampler.sample(100000)

    assert np.isclose(np.mean(samples), 0.5, atol=0.1)


# Distribution implementations for inverse transform sampling
def kumaraswamy_inverse_cdf(u, a, b):
    """Kumaraswamy inverse CDF - extremely simple closed form"""
    u = np.asarray(u)
    return (1 - (1 - u)**(1/b))**(1/a)


def kumaraswamy_cdf(x, a, b):
    """Kumaraswamy CDF for validation"""
    x = np.asarray(x)
    return 1 - (1 - x**a)**b


def power_inverse_cdf(u, k, alpha):
    """Power distribution - simplest possible inverse CDF"""
    u = np.asarray(u)
    return k * u**(1/alpha)


def triangular_inverse_cdf(u, a, b, c):
    """Triangular distribution with custom mode"""
    u = np.asarray(u)
    # Critical point where formula changes
    u_c = (c - a) / (b - a)
    
    # Vectorized implementation
    result = np.where(
        u <= u_c,
        a + np.sqrt(u * (b - a) * (c - a)),
        b - np.sqrt((1 - u) * (b - a) * (b - c))
    )
    return result


def rayleigh_inverse_cdf(u, sigma):
    """Rayleigh distribution - related to Maxwell-Boltzmann"""
    u = np.asarray(u)
    u = np.clip(u, 1e-15, 1-1e-15)  # Avoid log(0)
    return sigma * np.sqrt(-2 * np.log(1 - u))


def loglogistic_inverse_cdf(u, alpha, beta):
    """Log-logistic distribution - elegant closed form"""
    u = np.asarray(u)
    # Handle boundary cases to avoid division by zero
    u = np.clip(u, 1e-15, 1-1e-15)
    return alpha * (u / (1 - u))**(1/beta)


def loglogistic_cdf(x, alpha, beta):
    """Log-logistic CDF for validation"""
    x = np.asarray(x)
    return 1 / (1 + (alpha/x)**beta)


def gompertz_inverse_cdf(u, eta, b):
    """Gompertz distribution - reliability/survival analysis"""
    u = np.asarray(u)
    # Clip to avoid log(0)
    u = np.clip(u, 1e-15, 1-1e-15)
    return (1/b) * np.log(1 - np.log(1-u)/eta)


def reciprocal_inverse_cdf(u, a, b):
    """Reciprocal distribution - log-uniform"""
    u = np.asarray(u)
    return a * (b/a)**u


# Test cases for each distribution
def test_kumaraswamy_distribution():
    """Test Kumaraswamy distribution inverse transform sampling."""
    a, b = 2.0, 5.0
    n_samples = 10000
    
    # Create sampler
    sampler = create_sampler(
        inverse_cdf=lambda u: kumaraswamy_inverse_cdf(u, a, b),
        method="analytical",
        random_state=42,
    )
    
    # Generate samples
    samples = sampler.sample(n_samples)
    
    # Basic checks
    assert len(samples) == n_samples
    assert np.all(samples >= 0) and np.all(samples <= 1), "Samples should be in [0,1]"
    
    # Statistical validation using KS test against theoretical CDF
    def theoretical_cdf(x):
        return kumaraswamy_cdf(x, a, b)
    
    ks_stat, p_value = stats.kstest(samples, theoretical_cdf)
    assert p_value > 0.05, f"KS test failed with p-value: {p_value}"
    
    # Test quantile functionality
    quantiles = [0.1, 0.5, 0.9]
    quantile_values = sampler.sample_quantiles(quantiles)
    expected_values = kumaraswamy_inverse_cdf(quantiles, a, b)
    np.testing.assert_allclose(quantile_values, expected_values, rtol=1e-10)


def test_power_distribution():
    """Test Power distribution inverse transform sampling."""
    k, alpha = 3.0, 2.0
    n_samples = 10000
    
    # Create sampler
    sampler = create_sampler(
        inverse_cdf=lambda u: power_inverse_cdf(u, k, alpha),
        method="analytical",
        random_state=42,
    )
    
    # Generate samples
    samples = sampler.sample(n_samples)
    
    # Basic checks
    assert len(samples) == n_samples
    assert np.all(samples >= 0) and np.all(samples <= k), f"Samples should be in [0,{k}]"
    
    # Cross-validate against scipy.stats.powerlaw (different parameterization)
    # scipy uses: powerlaw(alpha, scale=k) equivalent to our power(k, alpha)
    scipy_samples = stats.powerlaw.rvs(alpha, scale=k, size=n_samples, random_state=42)
    ks_stat, p_value = stats.ks_2samp(samples, scipy_samples)
    assert p_value > 0.05, f"KS test against scipy failed with p-value: {p_value}"


def test_triangular_distribution():
    """Test Triangular distribution inverse transform sampling."""
    a, b, c = 1.0, 5.0, 3.0  # min, max, mode
    n_samples = 10000
    
    # Create sampler
    sampler = create_sampler(
        inverse_cdf=lambda u: triangular_inverse_cdf(u, a, b, c),
        method="analytical",
        random_state=42,
    )
    
    # Generate samples
    samples = sampler.sample(n_samples)
    
    # Basic checks
    assert len(samples) == n_samples
    assert np.all(samples >= a) and np.all(samples <= b), f"Samples should be in [{a},{b}]"
    
    # Cross-validate against scipy.stats.triang
    # scipy parameterization: triang(c_norm, loc=a, scale=b-a)
    c_norm = (c - a) / (b - a)
    scipy_samples = stats.triang.rvs(c_norm, loc=a, scale=b-a, size=n_samples, random_state=42)
    ks_stat, p_value = stats.ks_2samp(samples, scipy_samples)
    assert p_value > 0.05, f"KS test against scipy failed with p-value: {p_value}"
    
    # Test edge case: mode at left endpoint
    sampler_edge = create_sampler(
        inverse_cdf=lambda u: triangular_inverse_cdf(u, 0, 1, 0),
        method="analytical",
        random_state=43,
    )
    edge_samples = sampler_edge.sample(1000)
    assert np.all(edge_samples >= 0) and np.all(edge_samples <= 1)


def test_rayleigh_distribution():
    """Test Rayleigh distribution inverse transform sampling."""
    sigma = 2.0
    n_samples = 10000
    
    # Create sampler
    sampler = create_sampler(
        inverse_cdf=lambda u: rayleigh_inverse_cdf(u, sigma),
        method="analytical",
        random_state=42,
    )
    
    # Generate samples
    samples = sampler.sample(n_samples)
    
    # Basic checks
    assert len(samples) == n_samples
    assert np.all(samples >= 0), "Rayleigh samples should be non-negative"
    
    # Cross-validate against scipy.stats.rayleigh
    scipy_samples = stats.rayleigh.rvs(scale=sigma, size=n_samples, random_state=42)
    ks_stat, p_value = stats.ks_2samp(samples, scipy_samples)
    assert p_value > 0.05, f"KS test against scipy failed with p-value: {p_value}"
    
    # Check theoretical mean: σ√(π/2)
    theoretical_mean = sigma * np.sqrt(np.pi / 2)
    empirical_mean = np.mean(samples)
    assert np.isclose(empirical_mean, theoretical_mean, rtol=0.1), \
        f"Mean mismatch: empirical={empirical_mean:.3f}, theoretical={theoretical_mean:.3f}"


def test_loglogistic_distribution():
    """Test Log-logistic distribution inverse transform sampling."""
    alpha, beta = 2.0, 3.0
    n_samples = 10000
    
    # Create sampler
    sampler = create_sampler(
        inverse_cdf=lambda u: loglogistic_inverse_cdf(u, alpha, beta),
        method="analytical",
        random_state=42,
    )
    
    # Generate samples
    samples = sampler.sample(n_samples)
    
    # Basic checks
    assert len(samples) == n_samples
    assert np.all(samples > 0), "Log-logistic samples should be positive"
    
    # Validate against theoretical CDF
    def theoretical_cdf(x):
        return loglogistic_cdf(x, alpha, beta)
    
    ks_stat, p_value = stats.kstest(samples, theoretical_cdf)
    assert p_value > 0.05, f"KS test failed with p-value: {p_value}"
    
    # Test numerical stability near boundaries
    boundary_quantiles = [1e-10, 0.5, 1-1e-10]
    boundary_values = sampler.sample_quantiles(boundary_quantiles)
    assert np.all(np.isfinite(boundary_values)), "Boundary values should be finite"
    assert np.all(boundary_values > 0), "Log-logistic values should be positive"


def test_gompertz_distribution():
    """Test Gompertz distribution inverse transform sampling."""
    eta, b = 1.0, 0.5
    n_samples = 10000
    
    # Create sampler
    sampler = create_sampler(
        inverse_cdf=lambda u: gompertz_inverse_cdf(u, eta, b),
        method="analytical",
        random_state=42,
    )
    
    # Generate samples
    samples = sampler.sample(n_samples)
    
    # Basic checks
    assert len(samples) == n_samples
    assert np.all(samples >= 0), "Gompertz samples should be non-negative"
    assert np.all(np.isfinite(samples)), "All samples should be finite"
    
    # Validate using theoretical CDF since scipy parameterization is complex
    # For Gompertz: F(x) = 1 - exp(-eta * (exp(b*x) - 1))
    def gompertz_cdf(x, eta, b):
        return 1 - np.exp(-eta * (np.exp(b * x) - 1))
    
    def theoretical_cdf(x):
        return gompertz_cdf(x, eta, b)
    
    ks_stat, p_value = stats.kstest(samples, theoretical_cdf)
    assert p_value > 0.05, f"KS test against theoretical CDF failed with p-value: {p_value}"


def test_reciprocal_distribution():
    """Test Reciprocal (log-uniform) distribution inverse transform sampling."""
    a, b = 1.0, 10.0
    n_samples = 10000
    
    # Create sampler
    sampler = create_sampler(
        inverse_cdf=lambda u: reciprocal_inverse_cdf(u, a, b),
        method="analytical",
        random_state=42,
    )
    
    # Generate samples
    samples = sampler.sample(n_samples)
    
    # Basic checks
    assert len(samples) == n_samples
    assert np.all(samples >= a) and np.all(samples <= b), f"Samples should be in [{a},{b}]"
    
    # Cross-validate against scipy.stats.reciprocal
    scipy_samples = stats.reciprocal.rvs(a, b, size=n_samples, random_state=42)
    ks_stat, p_value = stats.ks_2samp(samples, scipy_samples)
    assert p_value > 0.05, f"KS test against scipy failed with p-value: {p_value}"
    
    # Test log-uniform property: log(samples) should be uniform
    log_samples = np.log(samples)
    log_expected_min, log_expected_max = np.log(a), np.log(b)
    
    # Check that log-transformed samples span expected range
    assert np.min(log_samples) >= log_expected_min - 0.1
    assert np.max(log_samples) <= log_expected_max + 0.1
    
    # KS test against uniform distribution in log space
    uniform_samples = stats.uniform.rvs(
        loc=log_expected_min, 
        scale=log_expected_max - log_expected_min,
        size=n_samples, 
        random_state=42
    )
    ks_stat_log, p_value_log = stats.ks_2samp(log_samples, uniform_samples)
    assert p_value_log > 0.05, f"Log-uniform test failed with p-value: {p_value_log}"
