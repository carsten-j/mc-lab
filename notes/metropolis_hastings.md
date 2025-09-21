# The Metropolis Hastings Algorithm: Theory and Convergence

## Table of Contents

1. [Introduction](#introduction)

2. [The General Algorithm](#the-general-algorithm)

3. [Theoretical Foundation](#theoretical-foundation)

4. [Convergence Theory](#convergence-theory)

5. [Spectral Analysis](#spectral-analysis)

6. [The Ergodic Theorem](#the-ergodic-theorem)

7. [Common Proposal Distributions](#common-proposal-distributions)

8. [Practical Considerations](#practical-considerations)

9. [Conclusion](#conclusion)

## Introduction

The Metropolis-Hastings (MH) algorithm is a Markov Chain Monte Carlo (MCMC) method for sampling from probability distributions that may be difficult to sample from directly. Originally developed by Metropolis et al. (1953) and generalized by Hastings (1970), it has become one of the most important computational tools in statistics, physics, and machine learning.

The fundamental idea is to construct a Markov chain whose stationary distribution is the target distribution $\pi(x)$. The algorithm achieves this through a clever accept/reject mechanism that ensures detailed balance, guaranteeing convergence to the desired distribution under appropriate conditions.

## The General Algorithm

### Problem Setup

Given a target distribution on $\mathbb{X} = \mathbb{R}^d$ with density $\pi(x)$ (known up to a normalization constant), we want to generate samples $X^{(1)}, X^{(2)}, \ldots$ such that for large $n$:

$$\frac{1}{n}\sum_{t=1}^n f(X^{(t)}) \approx \mathbb{E}_\pi[f(X)] = \int f(x)\pi(x)dx$$

### Algorithm Description

**Input:**

- Target density $\pi(x)$ (unnormalized)

- Proposal distribution $q(x'|x)$

- Starting value $X^{(1)}$

**For $t = 2, 3, \ldots$:**

1. **Proposal Step:** Sample $X^* \sim q(\cdot | X^{(t-1)})$

2. **Acceptance Probability:** Compute

$$\alpha(X^* | X^{(t-1)}) = \min\left(1, \frac{\pi(X^*) q(X^{(t-1)} | X^*)}{\pi(X^{(t-1)}) q(X^* | X^{(t-1)})}\right)$$

3. **Accept/Reject:** Sample $U \sim \text{Uniform}[0,1]$

- If $U \leq \alpha(X^* | X^{(t-1)})$, set $X^{(t)} = X^*$

- Otherwise, set $X^{(t)} = X^{(t-1)}$

### Key Properties

The acceptance probability $\alpha$ is the unique choice that:

- Ensures detailed balance with respect to $\pi(x)$

- Maximizes acceptance rates among all choices satisfying detailed balance

- Only requires $\pi(x)$ up to a proportionality constant

- Works for any valid proposal distribution $q(x'|x)$

## Theoretical Foundation

### Detailed Balance

The cornerstone of MH convergence is the **detailed balance condition**. For any two states $x, y \in \mathbb{X}$:

$$\pi(x)P(x \to y) = \pi(y)P(y \to x)$$

where $P(x \to y)$ is the transition probability from $x$ to $y$.

**Proof that MH satisfies detailed balance:**

For $x \neq y$, the transition probability is:

$$P(x \to y) = q(y|x)\alpha(x,y)$$

Using the MH acceptance probability:

$$\pi(x)P(x \to y) = \pi(x)q(y|x)\min\left(1, \frac{\pi(y)q(x|y)}{\pi(x)q(y|x)}\right)$$

Case 1: If $\frac{\pi(y)q(x|y)}{\pi(x)q(y|x)} \geq 1$, then $\alpha(x,y) = 1$:

$$\pi(x)P(x \to y) = \pi(x)q(y|x) = \pi(y)q(x|y) = \pi(y)P(y \to x)$$

Case 2: If $\frac{\pi(y)q(x|y)}{\pi(x)q(y|x)} < 1$, then $\alpha(x,y) = \frac{\pi(y)q(x|y)}{\pi(x)q(y|x)}$:

$$\pi(x)P(x \to y) = \pi(y)q(x|y) = \pi(y)P(y \to x)$$

In both cases, detailed balance holds.

### Stationary Distribution

When detailed balance is satisfied, $\pi$ becomes the stationary distribution:

$$\int \pi(x)P(x,dy) = \pi(y)$$

**Proof:** Integrating the detailed balance condition:

$$\int \pi(x)P(x \to y)dx = \int \pi(y)P(y \to x)dx = \pi(y)\int P(y \to x)dx = \pi(y)$$

### Transition Kernel

The complete transition kernel for the MH algorithm is:

$$P(x, dy) = q(y|x)\alpha(x,y)dy + \delta_x(dy)\left[1 - \int q(z|x)\alpha(x,z)dz\right]$$

The second term accounts for rejected proposals (remaining at the current state).

## Convergence Theory

### Ergodicity Conditions

For the MH chain to converge to $\pi$, it must be **ergodic**, requiring:

#### 1. Irreducibility

The chain can reach any set $A$ with $\pi(A) > 0$ from any starting point.

**Formal definition:** A chain is $\pi$-irreducible if for every $x \in \mathbb{X}$ and every set $A$ with $\pi(A) > 0$, there exists $n \geq 1$ such that $P^n(x,A) > 0$.

**For MH:** This requires that the proposal distribution eventually allows transitions to all regions of positive probability under $\pi$.

#### 2. Aperiodicity

The chain doesn't exhibit periodic behavior.

**For MH:** Usually satisfied automatically because there's always positive probability of staying at the current state due to potential rejections.

#### 3. Harris Recurrence

The chain returns to "large" sets infinitely often with probability 1.

### Types of Convergence

#### Geometric Ergodicity

A chain is geometrically ergodic if:

$$\|P^n(x,\cdot) - \pi(\cdot)\|_{TV} \leq M(x)\rho^n$$

for some $M(x) < \infty$ and $\rho < 1$.

**Sufficient conditions:** Often verified through drift conditions. There exists $\lambda < 1$, $b < \infty$, and a function $V(x) \geq 1$ such that:

$$PV(x) \leq \lambda V(x) + b \cdot \mathbf{1}_C(x)$$

for some petite set $C$.

#### Polynomial Ergodicity

When geometric ergodicity fails, the chain may still converge polynomially:

$$\|P^n(x,\cdot) - \pi(\cdot)\|_{TV} \leq M(x)n^{-\beta}$$

for some $\beta > 0$.

### Convergence Rate Bounds

For geometrically ergodic chains:

$$\|P^n(x,\cdot) - \pi(\cdot)\|_{TV} \leq 2\sqrt{\frac{\pi(x)^{-1}}{1-\rho}}\rho^{n/2}$$

The **mixing time** (time to reach within $\epsilon$ of stationarity) satisfies:

$$t_{mix}(\epsilon) \leq \frac{\log(\epsilon^{-1})}{\log(\rho^{-1})}$$

## Spectral Analysis

### The Transition Operator

The MH algorithm defines a transition operator $P$ acting on functions:

$$(Pf)(x) = \int f(y)P(x,dy)$$

For reversible chains, $P$ has real eigenvalues:

$$P\phi_i = \lambda_i \phi_i$$

with $1 = \lambda_0 > |\lambda_1| \geq |\lambda_2| \geq \cdots$

### Spectral Gap

The **spectral gap** is defined as:

$$\text{gap} = 1 - \lambda_1 = 1 - \max\{|\lambda_i| : i \geq 1\}$$

#### Connection to Convergence Rate

The convergence rate is directly related to the spectral gap:

$$\|P^n(x,\cdot) - \pi(\cdot)\|_2 \leq \lambda_1^n \|\delta_x - \pi\|_2$$

**Key insight:** Convergence is exponential with rate $\lambda_1 = 1 - \text{gap}$.

#### Mixing Time

$$t_{mix}(\epsilon) \approx \frac{\log(\epsilon^{-1})}{\text{gap}}$$

A larger spectral gap implies faster mixing.

### Cheeger's Inequality

For reversible chains, the spectral gap is bounded by the conductance $\Phi$:

$$\frac{\Phi^2}{2} \leq \text{gap} \leq 2\Phi$$

where the **conductance** is:

$$\Phi = \inf_{A: \pi(A) \leq 1/2} \frac{Q(A,A^c)}{\pi(A)}$$

and $Q(A,A^c) = \int_A \int_{A^c} \pi(x)P(x,dy)$ is the probability flux out of set $A$.

### Impact of Proposal Choice on Spectral Gap

#### Random Walk Metropolis

**Proposal:** $q(y|x) = N(y; x, \sigma^2 I)$

**Spectral gap scaling:**

- Optimal step size: $\sigma^2 \propto d^{-1}$ (where $d$ is dimension)

- Gap scales as: $\text{gap} \approx O(d^{-1})$

- Mixing time: $t_{mix} \approx O(d)$

**High-dimensional challenge:** The spectral gap shrinks as $O(d^{-1})$, leading to slow mixing in high dimensions.

#### Independence Metropolis

**Proposal:** $q(y|x) = g(y)$ (independent of current state)

**Spectral gap bound:**

$$\text{gap} \geq 1 - \sup_x \frac{\pi(x)}{g(x)}$$

**Key insight:** Gap depends on how well $g(x)$ approximates $\pi(x)$.

#### Langevin/HMC Methods

**Using gradient information:** $\nabla \log \pi(x)$

**Spectral gap:** Can achieve $\text{gap} \approx O(\kappa^{-1})$ where $\kappa$ is the condition number, potentially much better than $O(d^{-1})$.

## The Ergodic Theorem

### Strong Law of Large Numbers

For an ergodic MH chain $\{X^{(t)}\}$ with stationary distribution $\pi$:

**If $\mathbb{E}_\pi[|f(X)|] < \infty$, then:**

$$\lim_{n \to \infty} \frac{1}{n}\sum_{t=1}^n f(X^{(t)}) = \mathbb{E}_\pi[f(X)] \text{ almost surely}$$

**regardless of the starting state $X^{(1)}$.**

### Central Limit Theorem for MCMC

Under ergodicity and additional moment conditions:

$$\sqrt{n}\left(\bar{X}_n - \mu\right) \xrightarrow{d} N(0, \sigma_{eff}^2)$$

where:

- $\bar{X}_n = \frac{1}{n}\sum_{t=1}^n f(X^{(t)})$

- $\mu = \mathbb{E}_\pi[f(X)]$

- $\sigma_{eff}^2 = \text{Var}_\pi(f) + 2\sum_{k=1}^{\infty} \text{Cov}(f(X^{(1)}), f(X^{(1+k)}))$

### Effective Sample Size

The autocorrelation in MCMC samples affects efficiency:

$$\sigma_{eff}^2 = \sigma^2 \cdot \tau_{int}$$

where:

- $\sigma^2 = \text{Var}_\pi(f)$

- $\tau_{int} = 1 + 2\sum_{k=1}^{\infty} \rho_k$ (integrated autocorrelation time)

- $\rho_k = \text{Corr}(f(X^{(t)}), f(X^{(t+k)}))$

**Effective sample size:** $\text{ESS} = n/\tau_{int}$

### Conditions for Ergodic Theorem

The ergodic theorem applies when:

1. **Chain is ergodic** (irreducible + aperiodic + Harris recurrent)

2. **Function is integrable:** $\mathbb{E}_\pi[|f(X)|] < \infty$

### When the Ergodic Theorem Fails

#### Common Failure Modes

1. **Reducible chains:** Chain trapped in subset of state space

2. **Null recurrence:** Returns to sets too slowly

3. **Infinite expectations:** $\mathbb{E}_\pi[|f(X)|] = \infty$

## Common Proposal Distributions

### 1. Random Walk Proposals

#### Gaussian Random Walk

**Proposal:** $q(y|x) = N(y; x, \Sigma)$

**Algorithm:** Random Walk Metropolis-Hastings

Since $q(y|x) = q(x|y)$ (symmetric), the acceptance probability simplifies to:

$$\alpha(x,y) = \min\left(1, \frac{\pi(y)}{\pi(x)}\right)$$

**Characteristics:**

- Local exploration around current state

- Step size controlled by covariance $\Sigma$

- Good for unimodal, well-behaved distributions

- Can get stuck in local modes

**Optimal tuning:** For Gaussian targets in high dimensions:

- Step size: $\sigma^2 \approx 2.38^2 \lambda_{min}(\Sigma)/d$

- Target acceptance rate: ≈ 23%

#### Variants

- **Isotropic:** $\Sigma = \sigma^2 I$

- **Adaptive:** $\Sigma$ updated during burn-in

- **Preconditioned:** $\Sigma$ chosen to improve conditioning

### 2. Independence Proposals

#### Independence Sampler

**Proposal:** $q(y|x) = g(y)$ (independent of current state)

**Acceptance probability:**

$$\alpha(x,y) = \min\left(1, \frac{\pi(y)g(x)}{\pi(x)g(y)}\right)$$

**Characteristics:**

- Can make large jumps across state space

- Efficiency depends on quality of approximation $g(y) \approx \pi(y)$

- Good when reasonable approximation available

**Common choices for $g(y)$:**

- Multivariate normal approximation

- Mixture distributions

- Results from previous MCMC runs

### 3. Gibbs Sampling

#### Full Conditional Sampling

**Proposal:** $q(x_i^{new}|x) = \pi(x_i | x_{-i})$

**Acceptance probability:** $\alpha = 1$ (always accept)

**Algorithm:** Update one variable (or block) at a time using exact conditional distributions.

**Variants:**

- **Systematic scan:** Fixed update order

- **Random scan:** Random update order

- **Block Gibbs:** Update blocks jointly

### 4. Gradient-Based Proposals

#### Metropolis-Adjusted Langevin Algorithm (MALA)

**Proposal:**

$$q(y|x) = N\left(y; x + \frac{\epsilon^2}{2}\nabla \log \pi(x), \epsilon^2 I\right)$$

**Characteristics:**

- Uses gradient information to propose "uphill" moves

- Can achieve better scaling than random walk

- Requires differentiable target density

#### Hamiltonian Monte Carlo (HMC)

**Proposal:** Uses Hamiltonian dynamics with auxiliary momentum variables

**Characteristics:**

- Excellent performance for smooth, high-dimensional targets

- Requires gradient of log-density

- Can achieve dimension-independent performance

### 5. Adaptive Proposals

#### Adaptive Metropolis

**Proposal:** $q(y|x) = N(y; x, \Sigma_n)$ where $\Sigma_n$ adapts during sampling

**Roberts & Rosenthal Adaptive Metropolis:**

$$\Sigma_n = s_d[\text{Cov}(X_1, \ldots, X_{n-1}) + \epsilon I_d]$$

**Adaptation strategies:**

- Covariance matching

- Acceptance rate tuning

- Spectral gap optimization

## Practical Considerations

### Diagnostics and Monitoring

#### Convergence Diagnostics

1. **Trace plots:** Visual inspection of sample paths

2. **$\hat{R}$ statistic:** Compare multiple chains

3. **Autocorrelation plots:** Monitor correlation decay

4. **Effective sample size:** Measure efficiency

#### Multiple Chains

Run several chains from different starting points:

- Good mixing: chains converge to same distribution

- Poor mixing: chains remain separated

### Tuning Guidelines

#### Acceptance Rates

- **Random walk:** 20-50% (≈23% optimal for Gaussian targets)

- **Independence sampler:** Higher generally better

- **HMC:** 60-90%

#### Step Size Selection

- **Too small:** High acceptance, poor exploration

- **Too large:** Low acceptance, frequent rejections

- **Adaptive tuning:** Automatically adjust during burn-in

### Common Pitfalls

#### 1. Multimodal Distributions

**Problem:** Local proposals may not jump between modes

**Solutions:**

- Parallel tempering

- Independence samplers with good approximations

- Mixture proposals

#### 2. High Dimensions

**Problem:** Random walk mixing time grows as $O(d)$

**Solutions:**

- HMC/NUTS

- Block updates

- Dimension reduction

#### 3. Heavy Tails

**Problem:** Proposals may not explore tails adequately

**Solutions:**

- Heavy-tailed proposals

- Transformation to lighter tails

- Adaptive scaling

### Computational Considerations

#### Burn-in Period

Discard initial samples before reaching stationarity:

- Length depends on starting point and mixing rate

- Monitor convergence diagnostics

- Conservative: 10-50% of total samples

#### Thinning

Keep every $k$-th sample to reduce autocorrelation:

- Reduces storage requirements

- May improve effective sample size

- Not always necessary if post-processing accounts for correlation

## Implementation Best Practices

### Algorithm Structure

```python

def metropolis_hastings(target_log_density, proposal_sampler,

proposal_log_density, initial_state, n_samples):

"""

Generic Metropolis-Hastings implementation

"""

samples = []

current_state = initial_state

current_log_density = target_log_density(current_state)

for i in range(n_samples):

# Proposal step

proposed_state = proposal_sampler(current_state)

proposed_log_density = target_log_density(proposed_state)

# Acceptance probability (in log space for numerical stability)

log_alpha = (proposed_log_density +

proposal_log_density(current_state, proposed_state) -

current_log_density -

proposal_log_density(proposed_state, current_state))

# Accept/reject

if np.log(np.random.uniform()) < log_alpha:

current_state = proposed_state

current_log_density = proposed_log_density

samples.append(current_state.copy())

return np.array(samples)

```

### Numerical Stability

#### Log-Space Computations

Always work in log-space to avoid numerical overflow:

$$\log \alpha = \log \pi(y) + \log q(x|y) - \log \pi(x) - \log q(y|x)$$

#### Gradient Computations

For gradient-based methods, use automatic differentiation or finite differences with appropriate step sizes.

### Monitoring and Adaptation

#### Online Diagnostics

- Track acceptance rates

- Monitor sample statistics

- Detect poor mixing early

#### Adaptive Schemes

- Update proposal parameters during burn-in

- Use diminishing adaptation to preserve ergodicity

- Switch to fixed parameters for final sampling

## Advanced Topics

### Theoretical Extensions

#### Non-Reversible MCMC

Relaxing detailed balance while maintaining correct stationary distribution can improve mixing.

#### Infinite-Dimensional Targets

Extensions to function spaces (e.g., Gaussian processes, PDEs).

#### Tempering Methods

Use multiple chains at different "temperatures" to improve mixing for multimodal targets.

### Modern Developments

#### Hamiltonian Monte Carlo Variants

- NUTS (No-U-Turn Sampler)

- Riemannian HMC

- Relativistic HMC

#### Adaptive Methods

- Robust Adaptive Metropolis

- Population-based methods

- Machine learning-guided proposals

#### Parallel and Distributed MCMC

- Embarrassingly parallel chains

- Consensus-based methods

- Approximate methods for big data

## Conclusion

The Metropolis-Hastings algorithm represents a fundamental breakthrough in computational statistics, providing a general framework for sampling from complex probability distributions. Its theoretical foundation rests on three key pillars:

1. **Detailed Balance:** The acceptance probability ensures the target distribution becomes stationary

2. **Ergodicity:** Under appropriate conditions, the chain converges regardless of starting point

3. **Ergodic Theorem:** Time averages converge to ensemble averages, enabling Monte Carlo estimation

The algorithm's universality—working with any proposal distribution while maintaining convergence guarantees—makes it extraordinarily flexible. However, efficiency varies dramatically with proposal choice, highlighting the importance of understanding spectral properties and mixing behavior.

Key insights for practitioners:

- **Proposal design is crucial:** Match proposal to target geometry

- **Diagnostics are essential:** Always monitor convergence and mixing

- **Modern variants often superior:** HMC, NUTS, adaptive methods for many applications

- **Theory guides practice:** Spectral gap analysis explains performance differences

The Metropolis-Hastings algorithm continues to evolve, with ongoing research in adaptive methods, non-reversible chains, and specialized techniques for big data and complex models. Its combination of theoretical rigor and practical flexibility ensures its continued central role in computational statistics and machine learning.

### References

1. Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of state calculations by fast computing machines. *The Journal of Chemical Physics*, 21(6), 1087-1092.

2. Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 57(1), 97-109.

3. Roberts, G. O., & Rosenthal, J. S. (2004). General state space Markov chains and MCMC algorithms. *Probability Surveys*, 1, 20-71.

4. Meyn, S. P., & Tweedie, R. L. (2009). *Markov chains and stochastic stability*. Cambridge University Press.

5. Brooks, S., Gelman, A., Jones, G., & Meng, X. L. (Eds.). (2011). *Handbook of Markov chain Monte Carlo*. CRC press.
