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

# Detailed Analysis of Expected Squared Jump Distance (ESJD) and Optimal Scaling

## 1. Definition and Motivation of ESJD

### 1.1 What is ESJD?

The Expected Squared Jump Distance (ESJD) is a measure of efficiency for MCMC algorithms, defined as:

$$\text{ESJD} = E_\pi[\|X_{t+1} - X_t\|^2]$$

where the expectation is taken with respect to the stationary distribution $\pi$ and the transition kernel.

For a Metropolis-Hastings algorithm with proposal $Y$ given current state $X$:
$$\text{ESJD} = E_\pi[\|Y - X\|^2 \cdot \alpha(X, Y)]$$

where $\alpha(X, Y)$ is the acceptance probability.

**Why ESJD?** 
- It measures how far the chain moves per iteration on average
- Larger ESJD generally indicates faster mixing (though not always)
- It provides a scalar summary that's amenable to asymptotic analysis

## 2. ESJD for Random Walk Metropolis

### 2.1 Setting up the Calculation

For RWM with proposal $Y = X + \frac{\ell}{\sqrt{d}} Z$ where $Z \sim N(0, I_d)$:

$$\|Y - X\|^2 = \left\|\frac{\ell}{\sqrt{d}} Z\right\|^2 = \frac{\ell^2}{d} \|Z\|^2$$

Since $Z_i \sim N(0,1)$ independently, we have $\|Z\|^2 = \sum_{i=1}^d Z_i^2 \sim \chi^2(d)$.

### 2.2 Asymptotic Independence Result

A key insight for the asymptotic analysis is that as $d \to \infty$:
$$E_\pi[\|Y - X\|^2 \cdot \alpha(X, Y)] \approx E_\pi[\|Y - X\|^2] \cdot E_\pi[\alpha(X, Y)]$$

This asymptotic independence holds because:

1. **The jump distance** $\|Y - X\|^2 = \frac{\ell^2}{d}\|Z\|^2$ depends on $Z$
2. **The acceptance probability** $\alpha(X, Y)$ depends on $\log\frac{\pi(Y)}{\pi(X)} = -\frac{\ell}{\sqrt{d}} X \cdot Z - \frac{\ell^2}{2d}\|Z\|^2$

The key observation is that by the Law of Large Numbers:
$$\frac{1}{d}\|Z\|^2 \xrightarrow{p} 1 \quad \text{as } d \to \infty$$

So the variability in $\|Z\|^2$ becomes negligible relative to its mean, making it effectively constant.

### 2.3 Detailed Calculation of Each Component

**Component 1: Expected squared jump distance (unconditional)**

$$E[\|Y - X\|^2] = E\left[\frac{\ell^2}{d} \|Z\|^2\right] = \frac{\ell^2}{d} \cdot d = \ell^2$$

**Component 2: Expected acceptance probability**

From our earlier analysis, we showed that:
$$\log \frac{\pi(Y)}{\pi(X)} \xrightarrow{d} V \sim N\left(-\frac{\ell^2}{2}, \ell^2\right)$$

Therefore:
$$E[\alpha(X, Y)] \to \alpha(\ell) = 2\Phi\left(-\frac{\ell}{2}\right)$$

### 2.4 Combining the Results

As $d \to \infty$:
$$\text{ESJD} \approx \ell^2 \cdot 2\Phi\left(-\frac{\ell}{2}\right) = h(\ell)$$

## 3. Optimization of ESJD

### 3.1 Finding the Optimal $\ell$

We need to maximize:
$$h(\ell) = \ell^2 \cdot 2\Phi\left(-\frac{\ell}{2}\right)$$

Taking the derivative:
$$h'(\ell) = 2\ell \cdot 2\Phi\left(-\frac{\ell}{2}\right) + \ell^2 \cdot 2\phi\left(-\frac{\ell}{2}\right) \cdot \left(-\frac{1}{2}\right)$$

where $\phi$ is the standard normal PDF.

Setting $h'(\ell) = 0$:
$$4\ell\Phi\left(-\frac{\ell}{2}\right) = \ell^2 \phi\left(-\frac{\ell}{2}\right)$$

Dividing by $\ell$ (assuming $\ell > 0$):
$$4\Phi\left(-\frac{\ell}{2}\right) = \ell \phi\left(-\frac{\ell}{2}\right)$$

### 3.2 Numerical Solution

This equation doesn't have a closed-form solution, but numerical methods give:
- $\ell^* \approx 2.38$
- $\alpha(\ell^*) = 2\Phi(-\ell^*/2) \approx 0.234$

This is the famous 0.234 optimal acceptance rate for RWM in high dimensions!

### 3.3 Why This Specific Value?

The optimal value balances two competing effects:
1. **Large $\ell$**: Bigger proposed jumps ($\ell^2$ term grows)
2. **Small $\ell$**: Higher acceptance rate (but smaller jumps when accepted)

The value $\ell^* \approx 2.38$ optimally trades off these two effects.

## 4. More Rigorous Treatment of Asymptotic Independence

### 4.1 Why Can We Factor the Expectation?

The formal justification uses a concentration of measure argument. Define:
$$S = \frac{1}{d}\|Z\|^2$$

By the Law of Large Numbers and Central Limit Theorem:
- $E[S] = 1$
- $\text{Var}(S) = \frac{2}{d} \to 0$

So $S \xrightarrow{p} 1$. The acceptance probability can be written as:
$$\alpha(X, Y) = \min\left\{1, \exp\left(-\frac{\ell}{\sqrt{d}} X \cdot Z - \frac{\ell^2}{2}S\right)\right\}$$

Since $S \approx 1$ with high probability for large $d$:
$$\alpha(X, Y) \approx \min\left\{1, \exp\left(-\frac{\ell}{\sqrt{d}} X \cdot Z - \frac{\ell^2}{2}\right)\right\}$$

This depends on $Z$ only through $X \cdot Z/\sqrt{d}$, which is approximately independent of $\|Z\|^2/d$ for large $d$.

### 4.2 Error Terms

More precisely:
$$\text{ESJD} = \ell^2 \cdot 2\Phi\left(-\frac{\ell}{2}\right) + O(d^{-1/2})$$

The $O(d^{-1/2})$ error comes from:
1. The CLT approximation error: $O(d^{-1/2})$
2. The concentration of $\|Z\|^2/d$ around 1: $O(d^{-1/2})$
3. Higher-order correlations between jump size and acceptance

## 5. Comparison with MALA Scaling

### 5.1 MALA's Different Scaling

For MALA with $\sigma^2 \propto d^{-1/3}$:
$$\text{ESJD} \propto d^{-1/3} \cdot \alpha_{\text{MALA}}$$

where $\alpha_{\text{MALA}} \to 0.574$ optimally.

### 5.2 Efficiency Comparison

Per iteration:
- RWM: ESJD $\propto O(1)$ (constant in $d$)
- MALA: ESJD $\propto O(d^{-1/3})$

But we need to consider:
1. **Number of iterations to explore**: $\propto d/\text{ESJD}$
2. **RWM needs**: $O(d)$ iterations
3. **MALA needs**: $O(d^{4/3})$ iterations

However, MALA uses gradient information, which costs $O(d)$ evaluations per iteration.

## 6. Practical Implications

### 6.1 The 0.234 Rule

The result suggests tuning RWM to achieve approximately 23.4% acceptance rate. In practice:
- Start with some $\sigma^2$
- Monitor acceptance rate
- Adjust: if acceptance > 0.234, increase $\sigma^2$; if < 0.234, decrease $\sigma^2$

### 6.2 Limitations

The analysis assumes:
1. **Product form target**: May not hold for correlated targets
2. **Stationarity**: The chain has reached equilibrium
3. **High dimension**: Results are asymptotic in $d$
4. **Smoothness**: Target has required regularity

For finite $d$ or non-Gaussian targets, the optimal acceptance rate may differ from 0.234.

## 7. Extension to General Targets

For a general target $\pi$ with certain regularity conditions:
$$\text{ESJD} \approx \ell^2 \cdot 2\Phi\left(-\frac{I^{1/2} \ell}{2}\right)$$

where $I$ is the average Fisher information:
$$I = E_\pi\left[\frac{1}{d}\sum_{i=1}^d \left(\frac{\partial \log \pi}{\partial x_i}\right)^2\right]$$

The optimal acceptance rate remains approximately 0.234 for a broad class of targets.

# The 57% Optimal Acceptance Rate for MALA: Detailed Analysis

## 1. Overview of the MALA Scaling Result

The claim on page 9 states that with the correct scaling $\sigma^2 \asymp d^{-1/3}$, MALA achieves an optimal acceptance rate of approximately 57%. This is fundamentally different from RWM's 23.4% and requires a more delicate analysis.

## 2. The Key Difference: Order of the Remainder

### 2.1 Why MALA Behaves Differently

For MALA, the Metropolis log-ratio has a crucial property:
$$\Delta_d = \log \frac{\pi(Y)q(Y,X)}{\pi(X)q(X,Y)} = R_3^{(A)} + R_3^{(B)} = O(\|\eta\|^3)$$

The first and second-order terms cancel! This is not accidental but a consequence of MALA's design.

### 2.2 Detailed Expansion for Gaussian Target

For the Gaussian target with $\eta = Y - X = -\frac{\sigma^2}{2}X + \sigma Z$:

**Target ratio expansion:**
$$\log \pi(Y) - \log \pi(X) = -X^T\eta - \frac{1}{2}\|\eta\|^2 + \frac{1}{6}\sum_{i=1}^d \eta_i^3 + O(\|\eta\|^4)$$

**Proposal ratio expansion:**
The MALA proposals are:
- Forward: $q(x,y) \propto \exp\left(-\frac{\|y - x + \frac{\sigma^2}{2}x\|^2}{2\sigma^2}\right)$
- Reverse: $q(y,x) \propto \exp\left(-\frac{\|x - y + \frac{\sigma^2}{2}y\|^2}{2\sigma^2}\right)$

After careful expansion:
$$\log q(Y,X) - \log q(X,Y) = X^T\eta + \frac{1}{2}\|\eta\|^2 - \frac{1}{6}\sum_{i=1}^d \eta_i^3 + O(\|\eta\|^4)$$

The third-order term has the opposite sign! Therefore:
$$\Delta_d = \frac{1}{3}\sum_{i=1}^d \eta_i^3 + O(\|\eta\|^4)$$

## 3. Variance Calculation with $d^{-1/3}$ Scaling

### 3.1 Component Analysis

With $\sigma^2 = \ell^2 d^{-1/3}$:
- Each component: $\eta_i = -\frac{\sigma^2}{2}X_i + \sigma Z_i = O(\ell d^{-1/6})$
- Therefore: $\eta_i^3 = O(\ell^3 d^{-1/2})$

### 3.2 Distribution of $\Delta_d$

For the sum of third powers:
$$\Delta_d = \frac{1}{3}\sum_{i=1}^d \eta_i^3$$

Each $\eta_i^3$ has:
- Mean: $E[\eta_i^3] = E[(-\frac{\sigma^2}{2}X_i + \sigma Z_i)^3]$
- After expansion and using $E[X_i] = E[Z_i] = 0$, $E[X_i^2] = E[Z_i^2] = 1$:
  $$E[\eta_i^3] = -\frac{3\sigma^4}{2} + \sigma^3 \cdot 0 = -\frac{3\sigma^4}{2} = O(d^{-2/3})$$

The variance:
$$\text{Var}(\eta_i^3) = O(\sigma^6) = O(d^{-1})$$

Over $d$ dimensions:
$$\text{Var}(\Delta_d) = \frac{1}{9} \cdot d \cdot O(d^{-1}) = O(1)$$

## 4. The Limiting Distribution and Optimal Acceptance

### 4.1 Central Limit Theorem for $\Delta_d$

As $d \to \infty$ with $\sigma^2 = \ell^2 d^{-1/3}$:
$$\Delta_d \xrightarrow{d} W$$

where $W$ is a random variable with:
- Mean depending on $\ell$
- Variance proportional to $\ell^6$

### 4.2 The 57% Result

The optimal acceptance rate for MALA comes from maximizing:
$$\text{ESJD} \propto \sigma^2 \cdot P(\text{accept}) = \ell^2 d^{-1/3} \cdot E[\min\{1, e^{\Delta_d}\}]$$

Through detailed calculations (Roberts and Rosenthal, 1998), the optimization yields:
- Optimal $\ell^* \approx 1.65$
- Optimal acceptance rate $\approx 0.574$

### 4.3 Why 57% Instead of 23%?

The key differences from RWM:
1. **Better proposal**: MALA uses gradient information to propose moves toward higher probability regions
2. **Cancellation of lower-order terms**: The $O(\|\eta\|)$ and $O(\|\eta\|^2)$ terms cancel
3. **Smaller relative variance**: The acceptance probability has less variability

## 5. Heuristic Understanding

### 5.1 The Goldilocks Principle

For MALA:
- **Too small $\sigma$**: High acceptance (near 100%) but tiny moves
- **Too large $\sigma$**: Proposals overshoot, causing rejections
- **Just right**: 57% acceptance balances move size with acceptance

### 5.2 Information-Theoretic View

MALA uses gradient information more efficiently than RWM:
- RWM: Blind random walk, needs more rejections to "learn" the landscape
- MALA: Gradient guides proposals, can afford larger accepted moves

The 57% rate represents optimal information usage - enough rejections to prevent overshooting, enough acceptances to make progress.

## 6. Mathematical Details of the Optimization

### 6.1 The Optimization Problem

We maximize:
$$h(\ell) = \ell^2 \cdot E[\min\{1, \exp(W_\ell)\}]$$

where $W_\ell$ is the limiting distribution of $\Delta_d$ with parameter $\ell$.

### 6.2 First-Order Conditions

The derivative:
$$h'(\ell) = 2\ell \cdot E[\min\{1, e^{W_\ell}\}] + \ell^2 \cdot \frac{d}{d\ell}E[\min\{1, e^{W_\ell}\}]$$

Setting $h'(\ell^*) = 0$ and solving numerically gives $\ell^* \approx 1.65$.

### 6.3 The Acceptance Rate Formula

At the optimum:
$$\alpha_{\text{MALA}}^* = E[\min\{1, e^{W_{\ell^*}}\}] \approx 0.574$$

## 7. Robustness and Practical Considerations

### 7.1 Sensitivity to Dimension

The 57% rate is asymptotic. For finite $d$:
- $d < 50$: Optimal rate may be 50-70%
- $d = 100-1000$: Close to 57%
- $d > 1000$: Converges to 57%

### 7.2 Sensitivity to Target Distribution

For non-Gaussian targets:
- Log-concave targets: Similar optimal rate (55-60%)
- Heavy-tailed targets: Lower optimal rate
- Multi-modal targets: May not have a single optimal rate

### 7.3 Comparison of Optimal Rates

| Algorithm | Optimal Acceptance | Scaling | Efficiency |
|-----------|-------------------|---------|------------|
| RWM | 23.4% | $O(d)$ | Baseline |
| MALA | 57.4% | $O(d^{1/3})$ | $d^{2/3}$ times better |
| HMC | 65.1% | $O(d^{1/4})$ | $d^{3/4}$ times better |

## 8. Connection to Barker Proposals

The Barker proposal mentioned later achieves similar $O(d^{1/3})$ scaling but with different characteristics:
- More robust to tuning
- Slightly lower efficiency (by factor $\approx 2.47$)
- Better behavior in tails

The trade-off between MALA's 57% optimal rate (when perfectly tuned) versus Barker's robustness illustrates the efficiency-robustness trade-off central to the lecture.

## 9. Summary

The 57% optimal acceptance rate for MALA arises from:
1. The $O(\|\eta\|^3)$ scaling of the log-ratio
2. The $\sigma^2 \propto d^{-1/3}$ scaling requirement
3. Optimization of the expected squared jump distance
4. The balance between proposal quality and acceptance probability

This higher optimal acceptance rate compared to RWM (57% vs 23%) reflects MALA's more intelligent use of local geometry through gradient information, allowing it to propose larger moves that are more likely to be accepted.