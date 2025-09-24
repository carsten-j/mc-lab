# Markov Chain Monte Carlo: Local Balancing and Barker Proposals - Detailed Mathematical Analysis

## 1. Scaling Analysis of Random Walk Metropolis in High Dimensions

### 1.1 Problem Setup

Consider the target distribution in $\mathbb{R}^d$:
$$\pi(x) = \prod_{i=1}^{d} \varphi(x_i), \quad \text{where } \varphi(x_i) = \frac{1}{\sqrt{2\pi}} e^{-x_i^2/2}$$

The Random Walk Metropolis (RWM) proposal is:
$$Y = x + \frac{\ell}{\sqrt{d}} Z, \quad Z \sim N(0, I_d)$$

### 1.2 Detailed Derivation of the Log-Ratio

Given the product structure of $\pi$, we can write:
$$\log \frac{\pi(Y)}{\pi(X)} = \sum_{i=1}^{d} \log \frac{\varphi(Y_i)}{\varphi(X_i)} = \sum_{i=1}^{d} \left(-\frac{Y_i^2}{2} + \frac{X_i^2}{2}\right)$$

Since $Y_i = X_i + \frac{\ell}{\sqrt{d}} Z_i$, we have:
$$Y_i^2 = X_i^2 + 2X_i \cdot \frac{\ell}{\sqrt{d}} Z_i + \frac{\ell^2}{d} Z_i^2$$

Therefore:
$$\log \frac{\pi(Y)}{\pi(X)} = \sum_{i=1}^{d} \left(-X_i \cdot \frac{\ell}{\sqrt{d}} Z_i - \frac{\ell^2}{2d} Z_i^2\right) = -\frac{\ell}{\sqrt{d}} \sum_{i=1}^{d} X_i Z_i - \frac{\ell^2}{2d} \sum_{i=1}^{d} Z_i^2$$

This can be written compactly as:
$$\log \frac{\pi(Y)}{\pi(X)} = -\frac{\ell}{\sqrt{d}} X \cdot Z - \frac{\ell^2}{2d} \|Z\|^2$$

### 1.3 Central Limit Theorem Application

Under stationarity, $X \sim N(0, I_d)$ and independently $Z \sim N(0, I_d)$. We need to analyze the distribution of the log-ratio as $d \to \infty$.

**Step 1: Analyze the first term**
$$S_1 = -\frac{\ell}{\sqrt{d}} \sum_{i=1}^{d} X_i Z_i$$

Since $X_i$ and $Z_i$ are independent standard normal variables:
- $E[X_i Z_i] = 0$
- $\text{Var}(X_i Z_i) = E[X_i^2 Z_i^2] = E[X_i^2] E[Z_i^2] = 1$

Therefore:
- $E[S_1] = 0$
- $\text{Var}(S_1) = \frac{\ell^2}{d} \sum_{i=1}^{d} \text{Var}(X_i Z_i) = \frac{\ell^2}{d} \cdot d = \ell^2$

**Step 2: Analyze the second term**
$$S_2 = -\frac{\ell^2}{2d} \sum_{i=1}^{d} Z_i^2$$

Since $Z_i^2 \sim \chi^2(1)$:
- $E[Z_i^2] = 1$
- $\text{Var}(Z_i^2) = 2$

Therefore:
- $E[S_2] = -\frac{\ell^2}{2d} \cdot d = -\frac{\ell^2}{2}$
- $\text{Var}(S_2) = \frac{\ell^4}{4d^2} \cdot 2d = \frac{\ell^4}{2d} \to 0$ as $d \to \infty$

**Step 3: Apply CLT**

The sum $\sum_{i=1}^{d} X_i Z_i$ is a sum of i.i.d. random variables with mean 0 and variance 1. By the CLT:
$$\frac{1}{\sqrt{d}} \sum_{i=1}^{d} X_i Z_i \xrightarrow{d} N(0, 1)$$

For the second term, by the Law of Large Numbers:
$$\frac{1}{d} \sum_{i=1}^{d} Z_i^2 \xrightarrow{p} 1$$

**Combining the results:**
$$\log \frac{\pi(Y)}{\pi(X)} = -\ell \cdot N(0,1) - \frac{\ell^2}{2} + o_p(1) \xrightarrow{d} N\left(-\frac{\ell^2}{2}, \ell^2\right)$$

### 1.4 Derivation of the Limiting Acceptance Probability

The acceptance probability is:
$$\alpha(X, Y) = \min\left\{1, \exp\left(\log \frac{\pi(Y)}{\pi(X)}\right)\right\} = \min\{1, e^V\}$$

where $V \sim N(-\ell^2/2, \ell^2)$ in the limit.

**Calculating $E[\min\{1, e^V\}]$:**

$$\alpha(\ell) = E[1 \wedge e^V] = \int_{-\infty}^{\infty} \min\{1, e^v\} f_V(v) dv$$

where $f_V(v) = \frac{1}{\ell\sqrt{2\pi}} \exp\left(-\frac{(v + \ell^2/2)^2}{2\ell^2}\right)$.

Split the integral at $v = 0$:
$$\alpha(\ell) = \int_{-\infty}^{0} e^v f_V(v) dv + \int_{0}^{\infty} f_V(v) dv$$

**For the first integral:**
$$\int_{-\infty}^{0} e^v f_V(v) dv = \int_{-\infty}^{0} \frac{1}{\ell\sqrt{2\pi}} \exp\left(v - \frac{(v + \ell^2/2)^2}{2\ell^2}\right) dv$$

After completing the square in the exponent:
$$v - \frac{(v + \ell^2/2)^2}{2\ell^2} = -\frac{(v - \ell^2/2)^2}{2\ell^2}$$

This gives:
$$\int_{-\infty}^{0} e^v f_V(v) dv = \Phi\left(\frac{0 - \ell^2/2}{\ell}\right) = \Phi\left(-\frac{\ell}{2}\right)$$

**For the second integral:**
$$\int_{0}^{\infty} f_V(v) dv = P(V > 0) = P\left(\frac{V + \ell^2/2}{\ell} > \frac{\ell}{2}\right) = 1 - \Phi\left(\frac{\ell}{2}\right) = \Phi\left(-\frac{\ell}{2}\right)$$

Therefore:
$$\alpha(\ell) = 2\Phi\left(-\frac{\ell}{2}\right)$$

## 2. MALA Scaling Analysis

### 2.1 Taylor Expansion of the Log-Ratio

For MALA with proposal $Y = X + \eta$ where $\eta = -\frac{\sigma^2}{2}X + \sigma Z$:

**Expansion of $\log \pi(Y)$:**
$$\log \pi(Y) = \log \pi(X) + (\nabla \log \pi(X))^T \eta + \frac{1}{2} \eta^T \nabla^2 \log \pi(X) \eta + O(\|\eta\|^3)$$

For the Gaussian target, $\nabla \log \pi(X) = -X$ and $\nabla^2 \log \pi(X) = -I_d$, so:
$$\log \pi(Y) - \log \pi(X) = -X^T \eta - \frac{1}{2} \|\eta\|^2 + O(\|\eta\|^3)$$

**Expansion of proposal density ratio:**

The proposal densities are:
$$q(X, Y) = \frac{1}{(2\pi\sigma^2)^{d/2}} \exp\left(-\frac{\|Y - X - \frac{\sigma^2}{2}\nabla \log \pi(X)\|^2}{2\sigma^2}\right)$$

For our Gaussian case:
$$q(X, Y) = \frac{1}{(2\pi\sigma^2)^{d/2}} \exp\left(-\frac{\|Y - X + \frac{\sigma^2}{2}X\|^2}{2\sigma^2}\right)$$

The log proposal ratio is:
$$\log \frac{q(Y, X)}{q(X, Y)} = \frac{1}{2\sigma^2}\left(\|Y - X + \frac{\sigma^2}{2}X\|^2 - \|X - Y + \frac{\sigma^2}{2}Y\|^2\right)$$

After expansion (noting $\eta = Y - X$):
$$\log \frac{q(Y, X)}{q(X, Y)} = X^T \eta + \frac{1}{2}\|\eta\|^2 + O(\|\eta\|^3)$$

**Cancellation:** The Metropolis ratio becomes:
$$\Delta_d = \log \frac{\pi(Y)q(Y,X)}{\pi(X)q(X,Y)} = O(\|\eta\|^3)$$

### 2.2 Variance Analysis with $d^{-1/3}$ Scaling

With $\eta = -\frac{\sigma^2}{2}X + \sigma Z$ and $\sigma^2 = \ell^2 d^{-1/3}$:
- Each component: $\eta_i = O(\ell d^{-1/6})$
- Third moment: $E[\eta_i^3] = O(\ell^3 d^{-1/2})$
- Over $d$ dimensions with independence:
$$\text{Var}(\Delta_d) = O(d \cdot \ell^6 d^{-1}) = O(\ell^6) = O(1)$$

This ensures the acceptance probability remains bounded away from 0 and 1.

## 3. Local Balancing Theory

### 3.1 Proof of Local Balance Characterization

**Theorem:** $Q_g$ with $Q_g(x,y) = \frac{g(\pi(y)/\pi(x))K(x,y)}{Z_g(x)}$ is locally balanced if and only if $g(t) = t \cdot g(1/t)$.

**Proof Sketch:**

Local balance requires that as the step size $\sigma \to 0$:
$$\frac{\pi(x)Q_g(x,y)}{\sum_{z} \pi(z)Q_g(z,y)} \to \pi(x)$$

For small $\sigma$, the proposal concentrates near $x$. Using first-order approximation:
$$\frac{\pi(y)}{\pi(x)} \approx 1 + (\nabla \log \pi(x))^T(y-x)$$

The local balance condition becomes:
$$\lim_{\sigma \to 0} \frac{\pi(x) g(1 + \epsilon)}{\pi(x) g(1 + \epsilon) + \pi(y) g(1 - \epsilon)} = \frac{1}{2}$$

where $\epsilon = O(\sigma)$. This requires:
$$g(1 + \epsilon) = (1 - \epsilon) g(1 - \epsilon) + O(\epsilon^2)$$

Expanding $g$ around 1 and matching coefficients yields the functional equation $g(t) = t \cdot g(1/t)$.

### 3.2 Barker's Choice: $g(t) = t/(1+t)$

**Verification of local balance:**
$$g(1/t) = \frac{1/t}{1 + 1/t} = \frac{1}{t + 1}$$
$$t \cdot g(1/t) = \frac{t}{t + 1} = g(t) \checkmark$$

**Constant normalization:**

For the Barker choice:
$$Z_g(x) = \int g\left(e^{(\nabla \log \pi(x))^T(y-x)}\right) K(x,y) dy$$

With symmetric $K$ and using the substitution $z = -(y-x)$:
$$Z_g(x) = \int \frac{e^{(\nabla \log \pi(x))^T z}}{1 + e^{(\nabla \log \pi(x))^T z}} K(z) dz + \int \frac{1}{1 + e^{(\nabla \log \pi(x))^T z}} K(z) dz = \frac{1}{2}$$

## 4. Implementation Details for Barker Proposals

### 4.1 One-Dimensional Barker Algorithm

Given current state $x$ and gradient $g = \nabla \log \pi(x)$:

1. Sample $Z \sim N(0, \sigma^2)$
2. Compute acceptance probability for forward move:
   $$p_+ = \frac{1}{1 + \exp(-Zg)}$$
3. With probability $p_+$: propose $Y = x + Z$
4. With probability $1 - p_+$: propose $Y = x - Z$
5. Accept with Metropolis-Hastings probability:
   $$\alpha = \min\left\{1, \frac{\pi(Y)(R^2 + \|Y\|^2)^d}{\pi(X)(R^2 + \|X\|^2)^d} \cdot \frac{Q_B(Y,X)}{Q_B(X,Y)}\right\}$$

### 4.2 Coordinate-wise Barker in High Dimensions

For each coordinate $i = 1, \ldots, d$:
- Sample $Z_i \sim N(0, \sigma^2)$
- Compute $p_{+,i} = \frac{1}{1 + \exp(-Z_i \partial_i \log \pi(x))}$
- Set $Y_i = X_i + Z_i$ with probability $p_{+,i}$, else $Y_i = X_i - Z_i$

**Efficiency loss:** The factor of $15^{1/3} \approx 2.47$ comes from:
- Independent coordinate updates vs. joint updates
- Suboptimal use of gradient information
- Loss of rotational invariance

## 5. Stereographic and Rotational Variants

### 5.1 Stereographic Projection Mathematics

The stereographic projection $\text{SP}: \mathbb{R}^d \to S^d \setminus \{N\}$ is:
$$\text{SP}(x) = \left(\frac{2x}{R^2 + \|x\|^2}, \frac{R^2 - \|x\|^2}{R^2 + \|x\|^2}\right)$$

with inverse:
$$\text{SP}^{-1}(z) = \frac{R z_{1:d}}{1 - z_{d+1}}$$

The Jacobian determinant is:
$$\left|\det J_{\text{SP}}\right| = \left(\frac{2R}{R^2 + \|x\|^2}\right)^d$$

### 5.2 Givens Rotation Formula

Given orthonormal vectors $n_1, n_2$ and angle $\theta$:
$$R(\theta) = I_d + (n_2 n_1^T - n_1 n_2^T)\sin\theta + (n_1 n_1^T + n_2 n_2^T)[\cos\theta - 1]$$

This rotates vectors in the $(n_1, n_2)$-plane by angle $\theta$ while leaving the orthogonal complement fixed.

**Application to Barker:** Rotate to align gradient with $(1,1,\ldots,1)$ direction for optimal coordinate-wise performance.

## 6. Convergence and Ergodicity Properties

### 6.1 Geometric Ergodicity Conditions

A Markov chain is geometrically ergodic if there exists $\rho < 1$ and $C < \infty$ such that:
$$\|P^n(x, \cdot) - \pi(\cdot)\|_{TV} \leq C V(x) \rho^n$$

for some function $V \geq 1$.

**Heavy tails:** Methods fail geometric ergodicity when $\pi(x) \sim \|x\|^{-\alpha}$ for $\alpha$ insufficiently large.

**Super-light tails:** ULA becomes transient when $\pi$ decays faster than Gaussian, causing overshooting.

### 6.2 Robustness to Tuning

Barker proposals maintain stability because:
1. The acceptance function $g(t) = t/(1+t)$ is bounded
2. The proposal automatically reduces step size in low-probability regions
3. The local balance condition ensures correct limiting behavior

This contrasts with MALA where poor tuning can lead to:
- Persistent rejection (step size too large)
- Inefficient exploration (step size too small)
- Numerical instability in extreme regions

## Conclusion

The mathematical analysis reveals how local balancing provides a principled framework for constructing robust MCMC algorithms. The Barker proposal emerges as an optimal compromise between the efficiency of gradient-based methods and the robustness of simpler approaches, with stereographic variants offering additional theoretical guarantees for challenging target distributions.