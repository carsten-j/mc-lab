# Importance Sampling

ESS
https://www.tuananhle.co.uk/notes/ess.html
check the links to other resources

## Introduction

Importance sampling is a Monte Carlo technique that achieves dramatic variance reduction by strategically choosing where to sample from, rather than using uniform or natural sampling distributions. The core principle is to sample more frequently from regions that contribute most to the integral, then reweight the samples to maintain unbiased estimates.

## Motivation and Core Argument

### The Fundamental Problem with Ordinary Monte Carlo

In ordinary Monte Carlo integration, we estimate:

$$I=\int f(x)p(x) dx$$

using samples $X_1, X_2, \ldots, X_n \sim p(x)$ and the estimator:

$$\hat{I}_{MC}
=\frac{1}{n} \sum_{i=1}^n f(X_i)$$

The variance of this estimator is:

$$\text{Var}[\hat{I}_{MC}] = \frac{1}{n}\left[\int f^2(x)p(x)dx - I^2\right]$$

**The key issue**: Most samples may be "wasted" on regions where $f(x)p(x)$ is small, leading to high variance, especially for:

- Rare events (where $f(x)$ is an indicator function)
- Peaked integrands (where $f(x)$ is concentrated)
- Tail probabilities (where the important region has low probability under $p(x)$)

### The Importance Sampling Solution

Instead of sampling from the **nominal or target distribution** $p(x)$, we sample from a **importance or proposal distribution** $q(x)$ and use the estimator:

$$\hat{I}_{IS} = \frac{1}{n} \sum_{i=1}^n f(X_i)\frac{p(X_i)}{q(X_i)}$$​

where $X_i \sim q(x)$ and $\frac{p(X_i)}{q(X_i)}$​ are the **importance weights**.

The variance becomes:

$$\text{Var}[\hat{I}_{IS}] = \frac{1}{n}\left[\int \Big(\frac{f(x) p(x)}{q(x)}\Big)^2q(x)dx - I^2\right]$$

# Variance of the Importance Sampling Estimator

## Setup

In importance sampling, we want to estimate: $$I = \int f(x) p(x) dx = E_p[f(X)]$$

Instead of sampling from the target distribution $p(x)$, we sample from an importance distribution $q(x)$ and use the estimator: $$\hat{I} = \frac{1}{n} \sum_{i=1}^n f(X_i) \cdot w(X_i)$$

where $X_i \sim q(x)$ and $w(x) = \frac{p(x)}{q(x)}$ is the importance weight.

## Verifying Unbiasedness

Let's define $h(x) = f(x) \cdot w(x) = f(x) \cdot \frac{p(x)}{q(x)}$, so: $$\hat{I} = \frac{1}{n} \sum_{i=1}^n h(X_i)$$

The estimator is unbiased because: $$E_q[\hat{I}] = E_q[h(X)] = \int h(x) q(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \int f(x) p(x) dx = I$$

## Variance Derivation

Using the standard formula for the variance of a sample mean: $$\text{Var}_q(\hat{I}) = \frac{\text{Var}_q(h(X))}{n}$$

where: $$\text{Var}_q(h(X)) = E_q[h(X)^2] - (E_q[h(X)])^2 = E_q[h(X)^2] - I^2$$

Computing the second moment: $$E_q[h(X)^2] = E_q\left[\left(f(X) \cdot \frac{p(X)}{q(X)}\right)^2\right] = E_q\left[f(X)^2 \cdot \frac{p(X)^2}{q(X)^2}\right]$$

$$= \int f(x)^2 \cdot \frac{p(x)^2}{q(x)^2} \cdot q(x) dx = \int f(x)^2 \cdot \frac{p(x)^2}{q(x)} dx$$

## Final Result

The variance of the importance sampling estimator is:

$$\boxed{\text{Var}_q(\hat{I}) = \frac{1}{n} \left[ \int f(x)^2 \cdot \frac{p(x)^2}{q(x)} dx - I^2 \right]}$$

This can also be written as: $$\text{Var}_q(\hat{I}) = \frac{1}{n} \left[ E_q[f(X)^2 \cdot w(X)^2] - (E_q[f(X) \cdot w(X)])^2 \right]$$

## Key Properties

1. **Dependence on importance distribution**: The variance crucially depends on the choice of $q(x)$
2. **Optimal importance distribution**: The variance is minimized when: $$q^*(x) = \frac{|f(x)| p(x)}{\int |f(y)| p(y) dy}$$
3. **Comparison to standard MC**: For standard Monte Carlo ($q(x) = p(x)$), the variance is: $$\text{Var}_p(\hat{I}_{MC}) = \frac{1}{n} \left[ \int f(x)^2 p(x) dx - I^2 \right]$$
4. **Efficiency condition**: Importance sampling is more efficient when $q(x)$ places more probability mass where $f(x)^2 p(x)$ is large

The variance can be much smaller than standard Monte Carlo when $q(x)$ is well-chosen, but can also be much larger if $q(x)$ is poorly chosen (especially if $q(x)$ is small where $f(x)p(x)$ is large).
## Why Importance Sampling Achieves Dramatic Variance Reduction

### Theoretical Optimal Case

The variance-minimizing choice of proposal distribution is:

$$q^*(x) = \frac{|f(x)p(x)|}{\int|f(x)p(x)|dx}$$

**Remarkable result**: With this optimal choice, the variance becomes **exactly zero** when $f(x) \geq 0$:

- Each importance sampling estimate equals: $f(x_i)\frac{p(x_i)}{q^*(x_i)} = I$
- Every sample gives the true answer!

### Practical Variance Reduction

Even when we can't achieve the optimal $q^∗(x)$, getting close to it can yield:

- **1000×-10,000× variance reduction** for rare events
- **100×-1000× reduction** for peaked integrands
- **Orders of magnitude improvement** in computational efficiency

### Mathematical Analysis

The variance reduction factor is:

$$\text{Reduction Factor} = \frac{\int f^2(x)p(x)dx}{\int f^2(x)\frac{p^2(x)}{q(x)}dx}$$

**Massive reduction occurs when**:

1. $f(x)$ is concentrated in a small region
2. $q(x)$ is large where $f(x)p(x)$ is large
3. The ratio $\frac{p(x)}{q(x)}$ is relatively stable in important regions

### Concrete Example: Rare Event Estimation

**Problem**: Estimate $P(X>4)$ where $X \sim N(0,1)$

- True probability $\approx 3.17 \times 10^{-5}$

**Ordinary Monte Carlo**:

- Variance $\approx 3.17 \times 10^{-5}$
- Need $\sim 10^6$  samples for reasonable accuracy

**Importance Sampling with** $q(x) = N(4,1)$:

- Variance reduction factor >1000
- Good estimates with hundreds of samples

## Choosing the Proposal Distribution $q(x)$

### The Central Challenge

Since the optimal choice $q^*(x) = |f(x)p(x)|/\int|f(x)p(x)|dx$ requires knowing the integral we're computing, we need practical strategies.

### Practical Strategies

#### 1. Domain Knowledge Approach

- **Rare events**: Use heavy-tailed distributions (t-distribution, Pareto)
- **Peaked functions**: Center $q(x)$ around suspected peak locations
- **Tail probabilities**: Shift distribution toward tail regions

#### 2. Pilot Run Method

- Perform small ordinary Monte Carlo exploration
- Identify regions where $|f(x)p(x)|$ is large
- Fit parametric distribution to approximate the shape

#### 3. Heavy-Tailed Safety Net

- When uncertain, use distributions with heavier tails than $p(x)$
- Prevents catastrophic variance when $q(x)$ has lighter tails than $|f(x)p(x)|$

#### 4. Common Specific Choices

**For rare events** ($p(x) = N(0,1)$ , want $P(X>c)$ for large $c$):

$$q(x) = N(c, 1)$$

**For multimodal targets**:

$$q(x) = \sum_{i=1}^k w_i q_i(x)$$

### Critical Requirements

1. **Support condition**: $q(x) > 0$ wherever $f(x)p(x) \neq 0$ 
2. **Tail behavior**: $q(x)$ must have at least as heavy tails as $|f(x)p(x)|$
3. **Finite variance**: $\int f^2(x)\frac{p^2(x)}{q(x)}dx < \infty$

## Adaptive Methods for Proposal Distribution Selection

### The Fundamental Problem

Choosing good $q(x)$ requires knowing where the integrand is large, but we need good $q(x)$ to efficiently explore the integrand. Adaptive methods solve this chicken-and-egg problem through iterative learning.

### 1. Sequential Adaptive Importance Sampling

**Core Algorithm**:

```
Initialize: q₀(x), typically broad/conservative
For iteration t = 1, 2, ...:
  1. Sample x₁,...,xₙ ~ qₜ₋₁(x)
  2. Compute importance weights: wᵢ = f(xᵢ)p(xᵢ)/qₜ₋₁(xᵢ)
  3. Identify "important" regions (high |wᵢ|)
  4. Update qₜ(x) to increase probability in these regions
  5. Check convergence
```

**Key Insight**: Large importance weights indicate regions where current $q(x)$ is undersampling.

**Update Strategies**:

- **Parametric**: For $q_t(x) = N(\mu_t, \Sigma_t)$, update parameters based on weighted samples
- **Mixture**: Add components around high-weight regions
- **Kernel density**: Build $q_t(x)$ as weighted kernel density estimate

### 2. Population Monte Carlo (PMC)

Uses **multiple proposal distributions** simultaneously for robustness.

**Algorithm**:

```
Initialize: q₁⁽⁰⁾(x), ..., qₖ⁽⁰⁾(x), weights α₁⁽⁰⁾, ..., αₖ⁽⁰⁾
For iteration t = 1, 2, ...:
  1. Sample nᵢ points from each qᵢ⁽ᵗ⁻¹⁾(x)
  2. Compute importance weights for all samples
  3. Update mixture weights: αᵢ⁽ᵗ⁾ ∝ (effective sample size from qᵢ⁽ᵗ⁻¹⁾)
  4. Update component parameters based on weighted samples
  5. Optionally add/remove components
```

**Advantages**:

- Handles multimodal targets naturally
- More robust than single-distribution methods
- Automatically discovers multiple important regions

**Gaussian Component Update**:

$$\mu_i^{(t)} = \frac{\sum_j w_j^{(t)} \delta_{ij} x_j}{\sum_j w_j^{(t)} \delta_{ij}}$$

where $\delta_{ij} = 1$ if sample $j$ came from component $i$.

### 3. Cross-Entropy Method

Formulates finding good $q(x)$ as an **optimization problem**.

**Mathematical Foundation**:

- Optimal: $q^*(x) = |f(x)p(x)|/\int|f(x)p(x)|dx$ 
- Minimize cross-entropy: $CE(q^*,q) = -\int q^*(x)\log(q(x))dx$
- Equivalent to minimizing: $\int|f(x)p(x)|\log(q(x))dx$

**Algorithm**:

```
Initialize: q₀(x) with parameters θ₀
For iteration t = 1, 2, ...:
  1. Sample x₁,...,xₙ ~ qₜ₋₁(x)
  2. Compute importance weights: wᵢ = f(xᵢ)p(xᵢ)/qₜ₋₁(xᵢ)
  3. Select elite samples (top ρ% by |wᵢ|)
  4. Update θₜ by fitting qₜ(x) to elite samples
  5. Optional smoothing: θₜ = βθₜ + (1-β)θₜ₋₁
```

### 4. Adaptive Multiple Importance Sampling (AMIS)

Combines multiple importance sampling with adaptation.

**Key Features**:

- Maintains library of proposal distributions from all iterations
- Final estimator uses all past samples with multiple importance sampling weights
- Gradually builds mixture approximating $q^*(x)$

**Final Estimator**:

$$\hat{I} = \frac{\sum_t \sum_i w_{it} f(x_{it})}{\sum_t \sum_i w_{it}}$$

where $w_{it}$  are multiple importance sampling weights combining all proposals.

### 5. Modern Approaches

#### Neural Network Parameterization

- Use neural networks to represent $q(x; \theta)$
- Adapt $\theta$ using gradient-based optimization
- Powerful for high-dimensional problems

#### Gradient-Based Adaptation

- Directly optimize variance of IS estimator
- Use automatic differentiation for gradients w.r.t. $q(x)$ parameters

#### Variational Inference Connections

- Many adaptive IS methods relate to variational inference
- Can use VI techniques like natural gradients, ELBO optimization

## Practical Implementation Guidelines

### Starting Strategy

- Begin with heavy-tailed, broad distributions
- Use pilot runs for rough estimates of important regions

### Convergence Monitoring

- Track effective sample size: $ESS = \frac{(\sum w_i)^2}{\sum w_i^2}$
- Monitor variance estimates across iterations
- Check stability of $q(x)$ parameters

### Common Pitfalls

- **Premature convergence**: $q(x)$ converges to local optimum
- **Instability**: $q(x)$ changes too rapidly, loses important regions
- **Computational overhead**: Adaptation cost can be significant

### Balancing Exploration vs Exploitation

- Use smoothing: $q_t(x) = \alpha q_{adapted}(x) + (1-\alpha)q_{broad}(x)$
- Maintain probability mass in exploratory regions
- Apply regularization to prevent overfitting

## Method Selection Guidelines

|Method|Best For|Advantages|Disadvantages|
|---|---|---|---|
|Sequential Adaptive IS|Simple, unimodal targets|Easy to implement, low overhead|Can get stuck in local optima|
|Population Monte Carlo|Multimodal targets|Robust, handles complex structure|Higher computational cost|
|Cross-Entropy Method|Parametric families|Principled optimization|Requires good parameterization|
|Neural/Gradient-based|High-dimensional problems|Very flexible, powerful|Complex implementation, high cost|

## Conclusion

Importance sampling represents a fundamental shift from passive to active sampling in Monte Carlo methods. By intelligently choosing where to sample and appropriately reweighting, it can achieve:

- **Dramatic variance reduction** (1000× or more for rare events)
- **Computational efficiency gains** of orders of magnitude
- **Theoretical optimality** (zero variance in ideal case)

The key insights are:

1. **Concentrate samples** where the integrand is large
2. **Compensate through weights** to maintain unbiased estimates
3. **Adapt the strategy** when the optimal choice is unknown

Modern adaptive methods make importance sampling practical for complex, high-dimensional problems where the optimal proposal distribution is not obvious. The combination of theoretical foundation and practical adaptability makes importance sampling one of the most powerful tools in computational statistics and Monte Carlo methods.

