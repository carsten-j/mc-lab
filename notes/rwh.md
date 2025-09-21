# Metropolis-Hastings Algorithm: Convergence Theory and Proposal Design

## Table of Contents

1. [Algorithm Overview](#algorithm-overview)
2. [Convergence Theory](#convergence-theory)
3. [Ergodic Properties](#ergodic-properties)
4. [Proposal Distributions](#proposal-distributions)
5. [Optimal Scaling Theory](#optimal-scaling-theory)
6. [Advanced Proposal Strategies](#advanced-proposal-strategies)

---

## Algorithm Overview

The **Metropolis-Hastings (MH) algorithm** generates samples from a target distribution $\pi(x)$ by constructing a Markov chain with $\pi$ as its stationary distribution.

### Basic Algorithm

Given current state $x_t$:

1. **Propose**: Generate $y \sim q(y|x_t)$ from proposal distribution
2. **Accept/Reject**:
   - Compute acceptance probability:
   $$\alpha(x_t, y) = \min\left(1, \frac{\pi(y)q(x_t|y)}{\pi(x_t)q(y|x_t)}\right)$$
   - Set
   $$x_{t+1} = \begin{cases} y & \text{with probability } \alpha(x_t, y) \\ x_t & \text{otherwise} \end{cases}$$

---

## Convergence Theory

### Detailed Balance

The key to MH convergence is **detailed balance**: the transition kernel $P(x,y)$ satisfies:

$$\pi(x)P(x,y) = \pi(y)P(y,x) \quad \forall x,y$$

For MH, the transition probability is:
$$P(x,y) = \begin{cases}
q(y|x)\alpha(x,y) & \text{if } y \neq x \\
1 - \int q(z|x)\alpha(x,z)dz & \text{if } y = x
\end{cases}$$

### Proof of Detailed Balance

**Case 1**: $\frac{\pi(y)q(x|y)}{\pi(x)q(y|x)} \geq 1$

Then $\alpha(x,y) = 1$ and $\alpha(y,x) = \frac{\pi(x)q(y|x)}{\pi(y)q(x|y)}$

$$\pi(x)P(x,y) = \pi(x)q(y|x) = \pi(y)q(x|y) \cdot \frac{\pi(x)q(y|x)}{\pi(y)q(x|y)} = \pi(y)P(y,x)$$

**Case 2**: $\frac{\pi(y)q(x|y)}{\pi(x)q(y|x)} < 1$

Then $\alpha(x,y) = \frac{\pi(y)q(x|y)}{\pi(x)q(y|x)}$ and $\alpha(y,x) = 1$

$$\pi(x)P(x,y) = \pi(x)q(y|x) \cdot \frac{\pi(y)q(x|y)}{\pi(x)q(y|x)} = \pi(y)q(x|y) = \pi(y)P(y,x)$$

### Convergence Conditions

Under regularity conditions (irreducibility, aperiodicity, positive recurrence):

$$\lim_{n \to \infty} \|P^n(x, \cdot) - \pi\|_{TV} = 0$$

where $\|\cdot\|_{TV}$ is the total variation distance.

---

## Ergodic Properties

### 1. Standard Ergodicity

**Definition**: $\|P^n(x, \cdot) - \pi\|_{TV} \to 0$ as $n \to \infty$

**Conditions**:
- **Irreducible**: $\forall A, B$ with $\pi(A), \pi(B) > 0$, $\exists n$ such that $P^n(A,B) > 0$
- **Aperiodic**: $\gcd\{n : P^n(x,x) > 0\} = 1$
- **Positive recurrent**: $E_x[\tau_x] < \infty$ where $\tau_x$ is return time to $x$

### 2. Geometric Ergodicity

**Definition**: $\exists C, \rho < 1$ such that:
$$\|P^n(x, \cdot) - \pi\|_{TV} \leq C(x)\rho^n$$

**Sufficient Conditions** (Drift Condition):
$$\exists V: \mathbb{R}^d \to [1,\infty), \lambda < 1, b < \infty \text{ such that}$$
$$PV(x) \leq \lambda V(x) + b\mathbf{1}_C(x)$$

for some small set $C$.

**For Random Walk MH**: Often holds when target has sub-exponential tails:
$$\pi(x) \geq c\exp(-a\|x\|^{\alpha}) \text{ for some } \alpha < 2$$

### 3. Uniform Ergodicity

**Definition**:
$$\sup_x \|P^n(x, \cdot) - \pi\|_{TV} \leq C\rho^n$$

**Conditions**:
- Finite state space, or
- $\inf_{x \in S} \pi(x) > 0$ (bounded away from zero), or
- Very restrictive regularity conditions

---

## Proposal Distributions

### Types of Proposals

#### 1. Random Walk Proposals
$$q(y|x) = g(y-x)$$

**Gaussian**: $y = x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \Sigma)$

**Properties**:
- Symmetric: $q(y|x) = q(x|y)$
- Acceptance ratio: $\alpha(x,y) = \min\left(1, \frac{\pi(y)}{\pi(x)}\right)$
- Local exploration

#### 2. Independent Proposals
$$q(y|x) = q(y)$$

**Example**: $q(y) = \mathcal{N}(\mu, \Sigma)$

**Acceptance ratio**: $\alpha(x,y) = \min\left(1, \frac{\pi(y)q(x)}{\pi(x)q(y)}\right)$

#### 3. Gradient-Based Proposals (MALA)
$$y = x + \frac{\sigma^2}{2}\nabla \log \pi(x) + \sigma\varepsilon, \quad \varepsilon \sim \mathcal{N}(0,I)$$

**Properties**:
- Incorporates gradient information
- Better mixing than pure random walk
- Non-symmetric proposal

### Proposal Covariance Structures

#### Isotropic Proposals
$$\Sigma = \sigma^2 I$$

**Geometric interpretation**: Spherically symmetric proposals

$$\varepsilon \sim \mathcal{N}(0, \sigma^2 I) \Rightarrow p(\varepsilon) = (2\pi\sigma^2)^{-d/2}\exp\left(-\frac{\|\varepsilon\|^2}{2\sigma^2}\right)$$

**Matrix form**:
$$\sigma^2 I = \sigma^2 \begin{pmatrix}
1 & 0 & 0 & \cdots & 0 \\
0 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1
\end{pmatrix} = \begin{pmatrix}
\sigma^2 & 0 & 0 & \cdots & 0 \\
0 & \sigma^2 & 0 & \cdots & 0 \\
0 & 0 & \sigma^2 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \sigma^2
\end{pmatrix}$$

#### Diagonal (Non-Isotropic) Proposals
$$\Sigma = \text{diag}(\sigma_1^2, \sigma_2^2, \ldots, \sigma_d^2) = \begin{pmatrix}
\sigma_1^2 & 0 & 0 & \cdots & 0 \\
0 & \sigma_2^2 & 0 & \cdots & 0 \\
0 & 0 & \sigma_3^2 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \sigma_d^2
\end{pmatrix}$$

**Advantages**:
- Different step sizes per component
- Accounts for different scales across variables
- Still computationally efficient

#### General Covariance
$$\Sigma = \begin{pmatrix}
\sigma_1^2 & \sigma_{12} & \sigma_{13} & \cdots & \sigma_{1d} \\
\sigma_{12} & \sigma_2^2 & \sigma_{23} & \cdots & \sigma_{2d} \\
\sigma_{13} & \sigma_{23} & \sigma_3^2 & \cdots & \sigma_{3d} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\sigma_{1d} & \sigma_{2d} & \sigma_{3d} & \cdots & \sigma_d^2
\end{pmatrix}$$

**Correlation structure**: $\text{Corr}(X_i, X_j) = \frac{\sigma_{ij}}{\sigma_i\sigma_j}$

**2D Example**:
$$\Sigma = \begin{pmatrix}
\sigma_1^2 & \rho\sigma_1\sigma_2 \\
\rho\sigma_1\sigma_2 & \sigma_2^2
\end{pmatrix}$$

where $\rho \in (-1,1)$ is the correlation coefficient.

### Why Covariance Structure Matters

**Problem with isotropic proposals**: When target has different scales or correlations:

Target: $\pi(x,y) \propto \exp\left(-\frac{1}{2}\left[\left(\frac{x}{10}\right)^2 + \left(\frac{y}{0.1}\right)^2\right]\right)$

- $x$-direction: natural scale $\sim 10$
- $y$-direction: natural scale $\sim 0.1$

Using $\sigma^2 I$:
- If $\sigma$ small enough for $y$ → tiny steps in $x$
- If $\sigma$ large enough for $x$ → huge steps in $y$
- Result: poor mixing

**Better approach**: Match proposal to target structure:
$$\Sigma_{\text{proposal}} \propto \begin{pmatrix}
100 & 0 \\
0 & 0.01
\end{pmatrix}$$

---

## Optimal Scaling Theory

### Roberts-Gelman-Gilks Result (1997)

**Setting**: Random walk MH with $y = x + \sigma\varepsilon$, $\varepsilon \sim \mathcal{N}(0,I)$

**Target class**:
$$\pi(x_1,\ldots,x_d) \propto \exp\left(-\sum_{i=1}^d U(x_i)\right)$$

**Main Result**: As $d \to \infty$:
- **Optimal acceptance rate**: $2\Phi(-1) - 1 \approx 0.234$
- **Optimal step size**: $\sigma^* \propto d^{-1/2}$

where $\Phi$ is the standard normal CDF.

### Mathematical Derivation

The optimal step size $h$ maximizes the **speed measure**:
$$S(h) = h^2 \cdot \mathbb{E}[\min(1, \exp(-h|Z|))]$$

where $Z \sim \mathcal{N}(0,1)$.

**Speed measure breakdown**:
- $h^2$: squared step size (exploration per step)
- $\mathbb{E}[\min(1, \exp(-h|Z|))]$: expected acceptance probability

**Optimization**: Taking derivative and setting to zero:
$$\frac{dS}{dh} = 2h\mathbb{E}[\min(1, \exp(-h|Z|))] + h^2\mathbb{E}[\mathbf{1}_{|Z| \leq h}\cdot(-|Z|)\exp(-h|Z|)] = 0$$

**Solution**: $h^* \approx 1.59$, giving acceptance rate:
$$\mathbb{E}[\min(1, \exp(-1.59|Z|))] = 2\Phi(-1.59) \approx 0.234$$

### Extensions and Finite Dimension Results

**Roberts & Rosenthal (1998)**: Extended to more general targets

**Finite dimension behavior**:
- $d = 1$: optimal acceptance $\approx 44\%$
- $d = 5$: optimal acceptance $\approx 35\%$
- $d = 20$: optimal acceptance $\approx 25\%$
- $d \to \infty$: optimal acceptance $\to 23.4\%$

**Other algorithms**:
- **MALA**: optimal acceptance $\approx 57.4\%$ (Sherlock & Roberts, 2009)
- **HMC**: much higher optimal acceptance rates

### Practical Implications

**Tuning guidelines**:
- Target acceptance rate in range $[20\%, 50\%]$
- Lower end for higher dimensions
- Monitor effective sample size, not just acceptance rate

**Scaling with dimension**:
- Optimal step size $\sigma^* \propto d^{-1/2}$
- Mixing time grows polynomially with $d$
- Motivates more sophisticated algorithms (HMC, NUTS)

---

## Advanced Proposal Strategies

### Adaptive Covariance Methods

#### Robbins-Monro Adaptation
$$\Sigma_{n+1} = \Sigma_n + \gamma_n(\mathbf{x}_n\mathbf{x}_n^T - \Sigma_n)$$

where $\gamma_n$ is a decreasing step size sequence satisfying:
$$\sum_{n=1}^{\infty} \gamma_n = \infty, \quad \sum_{n=1}^{\infty} \gamma_n^2 < \infty$$

#### Empirical Covariance
$$\Sigma_n = \frac{1}{n-1}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}}_n)(\mathbf{x}_i - \bar{\mathbf{x}}_n)^T + \epsilon I$$

where $\bar{\mathbf{x}}_n = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i$ and $\epsilon > 0$ prevents singularity.

### Optimal Proposal Covariance

**Theoretical Result** (Roberts & Rosenthal): For many targets, optimal proposal covariance is:
$$\Sigma_{\text{optimal}} = \frac{(2.38)^2}{d} \Sigma_{\text{target}}$$

**Intuition**:
- Scale factor $(2.38)^2/d$ from optimal scaling theory
- Shape $\Sigma_{\text{target}}$ matches target geometry

### Block and Component Updates

#### Componentwise Updates
Update one component at a time:
$$x_i^{(t+1)} \sim q(x_i | x_{-i}^{(t)})$$

**Advantages**:
- Can tune each $\sigma_i^2$ separately
- Often better acceptance rates
- Simple to implement

**Disadvantages**:
- Can be slow for highly correlated targets
- Requires $d$ acceptance/rejection steps per iteration

#### Block Structure
For hierarchical parameters $\boldsymbol{\theta} = (\boldsymbol{\theta}_1, \boldsymbol{\theta}_2, \ldots, \boldsymbol{\theta}_k)$:

$$\Sigma = \begin{pmatrix}
\Sigma_1 & 0 & \cdots & 0 \\
0 & \Sigma_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \Sigma_k
\end{pmatrix}$$

**Use cases**:
- Natural parameter groupings
- Different scales across parameter types
- Computational efficiency

### Mixture Proposals

Combine multiple proposal types:
$$q(y|x) = \sum_{i=1}^k w_i q_i(y|x)$$

where $\sum_{i=1}^k w_i = 1$.

**Example**:
- $90\%$ small random walk steps: $q_1(y|x) = \mathcal{N}(x, \sigma_1^2 I)$
- $10\%$ large jumps: $q_2(y|x) = \mathcal{N}(x, \sigma_2^2 I)$ with $\sigma_2 \gg \sigma_1$

### Implementation Considerations

#### Numerical Stability
- **Positive definiteness**: Ensure $\Sigma \succ 0$
- **Cholesky decomposition**: $\Sigma = LL^T$ for sampling
- **Regularization**: $\Sigma + \epsilon I$ to prevent singularity

#### Computational Complexity
- **Isotropic** $\sigma^2I$: $O(d)$ storage, $O(d)$ sampling
- **Diagonal**: $O(d)$ storage, $O(d)$ sampling  
- **Full covariance**: $O(d^2)$ storage, $O(d^3)$ Cholesky factorization

#### Sampling from $\mathcal{N}(x, \Sigma)$
1. Compute Cholesky factor: $\Sigma = LL^T$
2. Generate $z \sim \mathcal{N}(0, I)$
3. Return $y = x + Lz$

---

## Practical Guidelines

### Tuning Strategy

1. **Start simple**: Begin with diagonal $\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$
2. **Tune acceptance**: Target 20-50% acceptance rate
3. **Monitor mixing**: Use effective sample size and trace plots
4. **Adapt covariance**: Learn correlation structure during burn-in
5. **Consider alternatives**: For high dimensions, consider HMC or NUTS

### Diagnostic Tools

**Acceptance rate**:
$$\hat{A} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}_{\{x_{i+1} \neq x_i\}}$$

**Effective sample size**: For chain $\{x_i\}_{i=1}^N$:
$$ESS = \frac{N}{1 + 2\sum_{k=1}^{\infty} \hat{\rho}_k}$$

where $\hat{\rho}_k$ is the lag-$k$ autocorrelation.

**Potential scale reduction factor** (Gelman-Rubin $\hat{R}$):
$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

where $\hat{V}$ is pooled variance estimate and $W$ is within-chain variance.

### Common Pitfalls

1. **Ignoring correlations**: Using $\sigma^2 I$ for correlated targets
2. **Poor step size**: Too large (low acceptance) or too small (slow mixing)
3. **Inadequate burn-in**: Starting diagnostics too early
4. **Single chain**: Not checking convergence across multiple chains
5. **Ignoring tail behavior**: Not accounting for heavy-tailed distributions

---

## Key Takeaways

1. **MH converges** through detailed balance to any target distribution
2. **Ergodic properties** depend on proposal choice and target regularity  
3. **Proposal design** is crucial - match target geometry when possible
4. **Optimal acceptance** $\approx 23.4\%$ for high-dimensional random walk (theoretical limit)
5. **General covariance** proposals significantly outperform isotropic for correlated targets
6. **Adaptive methods** can learn optimal proposal structure during sampling
7. **Computational trade-offs** exist between proposal sophistication and efficiency
8. **Modern alternatives** (HMC, NUTS) often superior for high-dimensional problems

---

## References

- Roberts, G.O., Gelman, A., & Gilks, W.R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *Annals of Applied Probability*, 7(1), 110-120.
- Roberts, G.O. & Rosenthal, J.S. (1998). Optimal scaling of discrete approximations to Langevin diffusions. *Journal of the Royal Statistical Society*, Series B, 60(1), 255-268.
- Sherlock, C. & Roberts, G.O. (2009). Optimal scaling of the random walk Metropolis on elliptically symmetric unimodal targets. *Bernoulli*, 15(3), 774-798.
- Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis algorithm. *Bernoulli*, 7(2), 223-242.
- Gelman, A., Gilks, W.R., & Roberts, G.O. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *Annals of Applied Probability*, 7(1), 110-120.
