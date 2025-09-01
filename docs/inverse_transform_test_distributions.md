# Distributions for Inverse Transform Sampling in Python

Inverse transform sampling requires distributions with closed-form inverse CDFs that aren't readily available in standard Python libraries. Here are the most suitable candidates for validating your implementation, prioritizing simplicity, mathematical interest, and testing value.

## 1. Kumaraswamy Distribution

**Parameters**: `a > 0` (shape), `b > 0` (shape)  
**Support**: (0, 1)

**Inverse CDF**: `F⁻¹(u) = (1 - (1 - u)^(1/b))^(1/a)`

**Python Implementation**:
```python
import numpy as np

def kumaraswamy_inverse_cdf(u, a, b):
    """Kumaraswamy inverse CDF - extremely simple closed form"""
    return (1 - (1 - u)**(1/b))**(1/a)

def kumaraswamy_pdf(x, a, b):
    """For validation purposes"""
    return a * b * x**(a-1) * (1 - x**a)**(b-1)

# Generate samples
uniform_samples = np.random.random(10000)
samples = kumaraswamy_inverse_cdf(uniform_samples, a=2, b=5)
```

**SciPy Status**: **Not available** - neither PDF/CDF nor sampling functions exist in scipy.stats.

**Verification Approach**:
```python
import scipy.stats as stats
from scipy.integrate import quad

# Validate using empirical CDF comparison
def validate_kumaraswamy(samples, a, b):
    # Theoretical CDF: F(x) = 1 - (1 - x^a)^b
    def theoretical_cdf(x):
        return 1 - (1 - x**a)**b
    
    # KS test against theoretical CDF
    ks_stat, p_value = stats.kstest(samples, theoretical_cdf)
    return ks_stat, p_value
```

**Special Considerations**: 
- Numerically stable for all parameter values
- Bounded to [0,1], excellent for testing boundary behavior
- Similar flexibility to Beta distribution but much simpler inverse CDF
- **Ideal first test case** due to simplicity

## 2. Power Distribution

**Parameters**: `k > 0` (scale), `α > 0` (shape)  
**Support**: (0, k)

**Inverse CDF**: `F⁻¹(u) = k × u^(1/α)`

**Python Implementation**:
```python
def power_inverse_cdf(u, k, alpha):
    """Power distribution - simplest possible inverse CDF"""
    return k * u**(1/alpha)

def power_pdf(x, k, alpha):
    """For validation"""
    return (alpha/k) * (x/k)**(alpha-1)

# Generate samples
samples = power_inverse_cdf(np.random.random(10000), k=3, alpha=2)
```

**SciPy Status**: **Available as powerlaw** with different parameterization, but inverse CDF not directly exposed.

**Verification Approach**:
```python
# Can validate against scipy.stats.powerlaw (note different parameterization)
def validate_power(samples, k, alpha):
    # Convert to scipy.stats.powerlaw parameterization
    scipy_samples = stats.powerlaw.rvs(alpha, scale=k, size=len(samples))
    ks_stat, p_value = stats.ks_2samp(samples, scipy_samples)
    return ks_stat, p_value
```

**Special Considerations**:
- Handles power-law behavior testing
- Scale parameter `k` tests domain transformation
- **Perfect for testing parameter sensitivity**

## 3. Triangular Distribution (Custom Mode)

**Parameters**: `a` (min), `b` (max), `c` (mode) where a ≤ c ≤ b  
**Support**: [a, b]

**Inverse CDF**:
```
F⁻¹(u) = a + √[u(b-a)(c-a)]           if u ≤ (c-a)/(b-a)
F⁻¹(u) = b - √[(1-u)(b-a)(b-c)]       if u > (c-a)/(b-a)
```

**Python Implementation**:
```python
def triangular_inverse_cdf(u, a, b, c):
    """Triangular distribution with custom mode"""
    # Critical point where formula changes
    u_c = (c - a) / (b - a)
    
    # Vectorized implementation
    result = np.where(
        u <= u_c,
        a + np.sqrt(u * (b - a) * (c - a)),
        b - np.sqrt((1 - u) * (b - a) * (b - c))
    )
    return result

def triangular_pdf(x, a, b, c):
    """For validation"""
    return np.where(
        x < c,
        2 * (x - a) / ((b - a) * (c - a)),
        2 * (b - x) / ((b - a) * (b - c))
    )
```

**SciPy Status**: **Available as triang**, but uses different parameterization and doesn't expose inverse CDF formula.

**Verification Approach**:
```python
def validate_triangular(samples, a, b, c):
    # Convert to scipy parameterization: triang(c_norm, loc=a, scale=b-a)
    c_norm = (c - a) / (b - a)
    scipy_samples = stats.triang.rvs(c_norm, loc=a, scale=b-a, size=len(samples))
    return stats.ks_2samp(samples, scipy_samples)
```

**Special Considerations**:
- Tests **piecewise inverse CDF** implementation
- Good for boundary condition testing (mode at endpoints)
- **Tests conditional logic** in inverse transform

## 4. Log-Logistic Distribution

**Parameters**: `α > 0` (scale), `β > 0` (shape)  
**Support**: (0, ∞)

**Inverse CDF**: `F⁻¹(u) = α × (u/(1-u))^(1/β)`

**Python Implementation**:
```python
def loglogistic_inverse_cdf(u, alpha, beta):
    """Log-logistic distribution - elegant closed form"""
    # Handle boundary cases to avoid division by zero
    u = np.clip(u, 1e-15, 1-1e-15)
    return alpha * (u / (1 - u))**(1/beta)

def loglogistic_pdf(x, alpha, beta):
    """For validation"""
    return (beta/alpha) * (x/alpha)**(beta-1) / (1 + (x/alpha)**beta)**2
```

**SciPy Status**: **Not directly available** - would need to be constructed as a transformed distribution.

**Verification Approach**:
```python
def validate_loglogistic(samples, alpha, beta):
    # Validate using theoretical moments
    if beta > 1:
        theoretical_mean = alpha * np.pi/beta / np.sin(np.pi/beta)
        empirical_mean = np.mean(samples)
        mean_error = abs(empirical_mean - theoretical_mean) / theoretical_mean
        
    # Custom CDF validation
    def theoretical_cdf(x):
        return 1 / (1 + (alpha/x)**beta)
    
    ks_stat, p_value = stats.kstest(samples, theoretical_cdf)
    return ks_stat, p_value, mean_error if beta > 1 else None
```

**Special Considerations**:
- **Unbounded distribution** with heavy tails
- Tests **numerical stability** near u=0 and u=1
- Related to logistic distribution (hence "log-logistic")

## 5. Gompertz Distribution

**Parameters**: `η > 0` (scale), `b > 0` (shape)  
**Support**: [0, ∞)

**Inverse CDF**: `F⁻¹(u) = (1/b) × ln(1 - (ln(1-u))/η)`

**Python Implementation**:
```python
def gompertz_inverse_cdf(u, eta, b):
    """Gompertz distribution - reliability/survival analysis"""
    # Clip to avoid log(0)
    u = np.clip(u, 1e-15, 1-1e-15)
    return (1/b) * np.log(1 - np.log(1-u)/eta)

def gompertz_pdf(x, eta, b):
    """For validation"""
    return b * eta * np.exp(b*x) * np.exp(-eta * (np.exp(b*x) - 1))
```

**SciPy Status**: **Available as gompertz** in scipy.stats, but **does not implement .rvs()** - falls back to slow PPF inversion.

**Verification Approach**:
```python
def validate_gompertz(samples, eta, b):
    # Validate against scipy PDF/CDF (but not sampling)
    scipy_dist = stats.gompertz(c=b, scale=1/eta)
    
    # KS test against scipy CDF
    ks_stat, p_value = stats.kstest(samples, scipy_dist.cdf)
    return ks_stat, p_value
```

**Special Considerations**:
- **Perfect example** of scipy distribution with CDF but no efficient sampling
- Used in reliability engineering and mortality modeling
- **Tests nested logarithms** and numerical precision

## 6. Reciprocal Distribution

**Parameters**: `a > 0` (lower bound), `b > a` (upper bound)  
**Support**: [a, b]

**Inverse CDF**: `F⁻¹(u) = a × (b/a)^u`

**Python Implementation**:
```python
def reciprocal_inverse_cdf(u, a, b):
    """Reciprocal distribution - log-uniform"""
    return a * (b/a)**u

def reciprocal_pdf(x, a, b):
    """For validation"""
    return 1 / (x * np.log(b/a))
```

**SciPy Status**: **Available as reciprocal** in scipy.stats but uses different parameterization and slow generic sampling.

**Verification Approach**:
```python
def validate_reciprocal(samples, a, b):
    # scipy uses different parameterization
    scipy_dist = stats.reciprocal(a, b)
    return stats.ks_2samp(samples, scipy_dist.rvs(size=len(samples)))
```

**Special Considerations**:
- **Log-uniform distribution** - excellent for testing log-scale phenomena
- Simple exponential formula tests **exponentiation accuracy**
- Bounded domain with multiplicative structure

## 7. Maxwell-Boltzmann Distribution

**Parameters**: `a > 0` (scale parameter)  
**Support**: [0, ∞)

**Inverse CDF**: No simple closed form, but **can be sampled** using `F⁻¹(u) ≈ a × √(-2 ln(1-u))` for the related **Rayleigh distribution** (Maxwell-Boltzmann is 3D, Rayleigh is 2D).

**Rayleigh Inverse CDF**: `F⁻¹(u) = σ × √(-2 ln(1-u))`

**Python Implementation**:
```python
def rayleigh_inverse_cdf(u, sigma):
    """Rayleigh distribution - related to Maxwell-Boltzmann"""
    u = np.clip(u, 1e-15, 1-1e-15)  # Avoid log(0)
    return sigma * np.sqrt(-2 * np.log(1 - u))

def rayleigh_pdf(x, sigma):
    """For validation"""
    return (x / sigma**2) * np.exp(-(x**2) / (2 * sigma**2))
```

**SciPy Status**: **Available as rayleigh** in scipy.stats with efficient sampling, making this good for **verification testing**.

**Special Considerations**:
- Tests **logarithm and square root** combinations
- **Physical interpretation** (molecular speeds)
- Can **cross-validate** against scipy implementation

## Comprehensive Validation Framework

Here's a complete validation framework for all distributions:

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class InverseTransformValidator:
    def __init__(self, inverse_cdf_func, pdf_func, domain):
        self.inverse_cdf = inverse_cdf_func
        self.pdf = pdf_func
        self.domain = domain
    
    def generate_samples(self, n_samples, **params):
        """Generate samples using inverse transform"""
        u = np.random.random(n_samples)
        return self.inverse_cdf(u, **params)
    
    def validate_samples(self, samples, significance_level=0.05):
        """Comprehensive validation suite"""
        results = {}
        
        # 1. KS test against theoretical CDF
        def theoretical_cdf(x):
            # Numerical integration of PDF
            from scipy.integrate import quad
            return quad(lambda t: self.pdf(t, **self.params), 
                       self.domain[0], x)[0]
        
        ks_stat, ks_p = stats.kstest(samples, theoretical_cdf)
        results['ks_test'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'passes': ks_p > significance_level
        }
        
        # 2. Basic statistical properties
        results['descriptive'] = {
            'mean': np.mean(samples),
            'std': np.std(samples),
            'min': np.min(samples),
            'max': np.max(samples),
            'n_samples': len(samples)
        }
        
        # 3. Visual validation
        self.plot_validation(samples)
        
        return results
    
    def plot_validation(self, samples):
        """Generate validation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram vs PDF
        axes[0,0].hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
        x = np.linspace(self.domain[0], self.domain[1], 1000)
        axes[0,0].plot(x, self.pdf(x, **self.params), 'r-', lw=2, label='Theoretical PDF')
        axes[0,0].legend()
        axes[0,0].set_title('PDF Comparison')
        
        # Q-Q plot
        stats.probplot(samples, dist=stats.norm, plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot')
        
        # Empirical CDF
        sorted_samples = np.sort(samples)
        empirical_cdf = np.arange(1, len(samples)+1) / len(samples)
        axes[1,0].plot(sorted_samples, empirical_cdf, label='Empirical CDF')
        axes[1,0].legend()
        axes[1,0].set_title('Empirical CDF')
        
        # Sample trace
        axes[1,1].plot(samples[:1000])
        axes[1,1].set_title('Sample Trace (first 1000)')
        
        plt.tight_layout()
        plt.show()

# Usage example
validator = InverseTransformValidator(
    inverse_cdf_func=kumaraswamy_inverse_cdf,
    pdf_func=kumaraswamy_pdf,
    domain=(0, 1)
)

samples = validator.generate_samples(10000, a=2, b=5)
validation_results = validator.validate_samples(samples)
```

## Implementation Priority and Testing Strategy

**Recommended Testing Order**:

1. **Kumaraswamy** - Simplest implementation, bounded domain
2. **Power Distribution** - Tests scaling and power operations
3. **Triangular** - Tests piecewise functions and conditional logic
4. **Rayleigh** - Tests logarithms and square roots, can cross-validate with scipy
5. **Log-Logistic** - Tests numerical stability and unbounded domains
6. **Gompertz** - Tests complex nested functions
7. **Reciprocal** - Tests exponential operations and log-uniform behavior

**Critical Test Cases**:
- **Boundary values**: u = 0, u = 1, u = 0.5
- **Extreme parameters**: Very small/large shape parameters
- **Large samples**: n = 10⁶ to test computational efficiency
- **Numerical precision**: Compare double vs. single precision results

This comprehensive set provides excellent coverage of different mathematical structures, numerical challenges, and validation approaches for testing inverse transform sampling implementations.