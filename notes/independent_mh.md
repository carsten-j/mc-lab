# Independent Metropolis-Hastings Algorithm: Theory and Practice

## Table of Contents

1. [Introduction](#introduction)

2. [Convergence Theory](#convergence-theory)

3. [Proposal Distribution Guidelines](#proposal-distribution-guidelines)

4. [Adaptive Approaches](#adaptive-approaches)

5. [Importance Sampling Connection](#importance-sampling-connection)

6. [Autocorrelation Time and Diagnostics](#autocorrelation-time-and-diagnostics)

7. [Practical Implementation](#practical-implementation)

8. [Trade-offs and Design Principles](#trade-offs-and-design-principles)

## Introduction

The independent Metropolis-Hastings algorithm is a variant of the Metropolis-Hastings algorithm where the proposal distribution does not depend on the current state. Given a target distribution $\pi(x)$ that we want to sample from, the algorithm works as follows:

### Algorithm Structure

At current state $x$, we:

1. Propose $x' \sim q(x')$ (independent of $x$)

2. Accept with probability $\alpha(x,x') = \min\left(1, \frac{\pi(x')q(x)}{\pi(x)q(x')}\right)$

3. Set $X^{(t+1)} = x'$ if accepted, otherwise $X^{(t+1)} = X^{(t)}$

Where $\pi$ is our target distribution and $q$ is the proposal distribution.

## Convergence Theory

### Why the Algorithm Converges

The independent Metropolis-Hastings algorithm converges to the target distribution due to fundamental properties of Markov chains:

#### 1. Detailed Balance Condition

The algorithm satisfies detailed balance: $\pi(x)P(x,x') = \pi(x')P(x',x)$, where $P(x,x')$ is the transition probability.

For the transition probability $P(x,x') = q(x')\alpha(x,x')$, we need to verify:

$$\pi(x)q(x')\alpha(x,x') = \pi(x')q(x)\alpha(x',x)$$

This holds because of the construction of the acceptance probability. When $\pi(x')q(x) \geq \pi(x)q(x')$:

- $\alpha(x,x') = 1$

- $\alpha(x',x) = \frac{\pi(x)q(x')}{\pi(x')q(x)}$

Making both sides equal $\pi(x)q(x')$.

#### 2. Irreducibility

If the proposal distribution $q(x')$ has positive probability wherever $\pi(x) > 0$, then any state can eventually reach any other state. This condition is:

$$\text{supp}(\pi) \subseteq \text{supp}(q)$$

#### 3. Aperiodicity

The chain is typically aperiodic because there's always positive probability of staying in the same state when proposals are rejected.

### Ergodicity Hierarchy

There are different types of ergodicity that describe convergence rates:

#### Basic Ergodicity

$$\lim_{n\to\infty} \|P^n(x,\cdot) - \pi(\cdot)\|_{TV} = 0$$

#### Geometric Ergodicity

$$\|P^n(x,\cdot) - \pi(\cdot)\|_{TV} \leq M(x)\rho^n$$

where $0 < \rho < 1$ and $M(x) < \infty$.

#### Uniform Ergodicity

$$\|P^n(x,\cdot) - \pi(\cdot)\|_{TV} \leq M\rho^n$$

where $M$ is independent of $x$.

#### Heavy Tail Condition

For geometric ergodicity, we need:

$$\sup_x \frac{\pi(x)}{q(x)} < \infty$$

When this ratio is unbounded, the chain loses geometric ergodicity and may have only polynomial convergence.

## Proposal Distribution Guidelines

### Fundamental Requirements

#### 1. Support Condition

$$q(x') > 0 \text{ wherever } \pi(x) > 0$$

#### 2. Heavy Tail Condition

$$\sup_x \frac{\pi(x)}{q(x)} < \infty$$

### Practical Design Strategies

#### 3. Shape Approximation

Choose $q$ to roughly approximate $\pi$'s shape and location:

- Normal distribution: $q(x) = \mathcal{N}(\hat{\mu}, \hat{\Sigma})$ where $\hat{\mu}$ and $\hat{\Sigma}$ are estimates of target moments

#### 4. Tail Behavior Matching

- If $\pi$ has exponential tails, $q$ should have exponential or heavier tails

- If $\pi$ has polynomial tails, $q$ needs polynomial tails of same or higher degree

#### 5. Mixture Proposals

$$q(x') = \sum_i w_i q_i(x')$$

Typical structure:

$$q(x') = 0.9 \times \mathcal{N}(\hat{\mu}, \hat{\Sigma}) + 0.1 \times t_\nu(\hat{\mu}, \hat{\Sigma})$$

### Performance Metrics

Monitor these diagnostics:

- **Acceptance rate**: Typically want 20-50% for independent MH

- **Effective sample size**: $\text{ESS} = \frac{N}{2\tau + 1}$

- **Autocorrelation time**: $\tau = 1 + 2\sum_{k=1}^{\infty} \rho(k)$

## Adaptive Approaches

### Three-Phase Strategy

#### Phase 1: Initial Approximation

Start with:

$$q_0(x) = \mathcal{N}(\mu_0, \Sigma_0)$$

where:

- $\mu_0 = \arg\max \pi(x)$ (approximate mode)

- $\Sigma_0 = c \times I$ (conservative initial variance)

#### Phase 2: Pilot Runs

```

for k = 1, 2, ..., K:

Run chain for n_pilot iterations with current q_k

Compute sample statistics

Update q_{k+1} based on samples

```

Track during pilots:

- Sample mean: $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$

- Sample covariance: $\hat{\Sigma} = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^T$

- Acceptance rate

- Effective sample size

#### Phase 3: Iterative Improvement

**Moment Matching Update:**

$$\mu_{\text{new}} = \bar{x}_{\text{pilot}}$$

$$\Sigma_{\text{new}} = \alpha \hat{\Sigma}_{\text{empirical}} + (1-\alpha) \Sigma_{\text{current}}$$

**Adaptive Scaling:**

```

if acceptance_rate < 0.2:

Σ_new = 0.8 * Σ_current # Shrink proposal

elif acceptance_rate > 0.5:

Σ_new = 1.2 * Σ_current # Expand proposal

```

### Convergence Criteria

**KL Divergence Test:**

$$D_{KL}(q_{\text{old}} \| q_{\text{new}}) < \epsilon$$

**Chain Convergence:**

- Gelman-Rubin statistic: $\hat{R} < 1.1$

- Effective sample size: $\text{ESS} > \text{ESS}_{\min}$

## Importance Sampling Connection

### The Connection

The acceptance rate in independent MH equals the harmonic mean of importance weights:

$$\text{Acceptance Rate} = \frac{1}{\mathbb{E}_q[\pi(X)/q(X)]} = \frac{1}{\int \frac{\pi(x)}{q(x)} q(x) dx}$$

### Variance Minimization

The importance weights are $w(x) = \pi(x)/q(x)$. We want to minimize:

$$\text{Var}_q[w(X)] = \mathbb{E}_q[w(X)^2] - (\mathbb{E}_q[w(X)])^2$$

**Theoretical Optimum:**

The variance-minimizing proposal is $q^*(x) \propto \pi(x)$ (the target itself), but this is circular.

### Practical Strategies

#### 1. Weight Monitoring

```python

diagnostics = {

'cv_weights': np.std(weights) / np.mean(weights),

'effective_sample_size': N / (1 + np.var(weights)/np.mean(weights)**2),

'max_weight_ratio': np.max(weights) / np.mean(weights)

}

```

#### 2. Red Flags

- Coefficient of variation > 2

- ESS < 10% of sample size

- Maximum weight > 100× mean weight

#### 3. Defensive Mixture

$$q(x) = 0.9 \times q_{\text{main}}(x) + 0.1 \times q_{\text{heavy-tail}}(x)$$

## Autocorrelation Time and Diagnostics

### Mathematical Definition

**Autocorrelation Function:**

$$\rho(k) = \frac{\mathbb{E}[(X_t - \mu)(X_{t+k} - \mu)]}{\text{Var}(X)}$$

**Sample Estimator:**

$$\hat{\rho}(k) = \frac{\sum_{i=1}^{n-k} (X_i - \bar{X})(X_{i+k} - \bar{X})}{\sum_{i=1}^n (X_i - \bar{X})^2}$$

**Integrated Autocorrelation Time:**

$$\tau = 1 + 2\sum_{k=1}^{\infty} \rho(k)$$

### Practical Implementation

```python

def estimate_acf(samples, max_lag=None):

n = len(samples)

if max_lag is None:

max_lag = min(n//4, 200)

samples_centered = samples - np.mean(samples)

autocorr = np.zeros(max_lag + 1)

variance = np.var(samples)

for k in range(max_lag + 1):

if k == 0:

autocorr[k] = 1.0

else:

valid_length = n - k

autocorr[k] = np.sum(samples_centered[:-k] * samples_centered[k:]) / (valid_length * variance)

return autocorr

```

### Relationship to Effective Sample Size

$$\text{ESS} = \frac{N}{2\tau + 1}$$

### Monte Carlo Standard Error

$$\text{MCSE} = \sigma\sqrt{\frac{2\tau}{N}}$$

where $\sigma$ is the marginal standard deviation.

## Practical Implementation

### Complete Algorithm Template

```python

def adaptive_independent_mh(target_log_pdf, initial_q, max_pilots=10):

q_current = initial_q

for pilot in range(max_pilots):

# Run pilot chain

samples, acceptance_rate = run_chain(

target_log_pdf, q_current, n_iter=pilot_length[pilot]

)

# Check convergence

if chain_converged(samples) and proposal_converged(q_old, q_current):

break

# Update proposal

q_new = update_proposal(q_current, samples)

q_current = q_new

return q_current, final_chain_run(target_log_pdf, q_current)

```

### Diagnostic Dashboard

```python

def diagnose_chain(samples, target_log_pdf, proposal):

weights = np.exp(target_log_pdf(samples) - proposal.logpdf(samples))

diagnostics = {

'acceptance_rate': compute_acceptance_rate(samples),

'autocorr_time': max_autocorr_time(samples),

'cv_weights': np.std(weights) / np.mean(weights),

'ess': effective_sample_size(samples),

'gelman_rubin': gelman_rubin_statistic(samples)

}

return diagnostics

```

## Trade-offs and Design Principles

### The Fundamental Trade-off

There is an inherent tension between:

- **Large moves**: Better exploration, lower correlation

- **High acceptance**: Efficient use of computational budget

### Sources of Correlation

1. **Proposal dependence**: $q(\cdot|X^{(t-1)})$ - eliminated in independent MH

2. **Rejection-induced correlation**: When $X^{(t)} = X^{(t-1)}$ due to rejection

### Covariance Structure Matching

**Critical Requirement**: For multivariate targets, the proposal covariance should reflect the target's correlation structure:

$$\Sigma_{\text{proposal}} \approx c \times \Sigma_{\text{target}}$$

where $c > 1$ is an inflation factor for exploration.

**Consequences of Mismatch**:

- Chain moves inefficiently along correlation directions

- Dramatically increased autocorrelation time

- Poor exploration of the target distribution

### Comparison with Other Methods

| Method | Pros | Cons |

|--------|------|------|

| Independent MH | Simple, eliminates proposal dependence | Requires good global approximation |

| Random Walk MH | Adapts locally, easier to tune | Can get trapped, high correlation |

| Hamiltonian MC | Uses gradients, excellent performance | Requires differentiable target |

### Best Practices Summary

1. **Start conservatively**: Use heavy-tailed initial proposals

2. **Monitor diagnostics**: Track acceptance rate, autocorrelation time, ESS

3. **Adapt systematically**: Use pilot runs with decreasing adaptation rates

4. **Match covariance structure**: Essential for multivariate targets

5. **Ensure heavy tails**: Maintain $\sup_x \pi(x)/q(x) < \infty$

6. **Use defensive mixtures**: Combine target approximation with exploration components

The independent Metropolis-Hastings algorithm provides a powerful framework for MCMC sampling when the proposal distribution can be designed to globally approximate the target distribution. Success depends critically on balancing exploration capabilities (heavy tails) with efficiency (shape matching), while ensuring the theoretical conditions for convergence are met.
