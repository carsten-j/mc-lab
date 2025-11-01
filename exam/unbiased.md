## Motivation & Intro

MCMC estimator are **consistent** (under ergodicity conditions) as $t \rightarrow \infty$, but **potentially biased for a fixed number of iterations**. 
Usually we deal with this by splitting the chain in two phases:

- **Mixing/transient phase**: were the chain moves from its initial point to the distribution it targets
- **Stationary phase**: once the chain has converged to its invariant distribution we use the sample coming from the chain in a Monte Carlo estimator.

Since we lack good diagnostic of convergence, and burn-in period is somewhat a waste of computation, we would like to been able to also use the burn-in samples without introducing bias in our estimates. As the name suggests, this is the goal of an unbiased MCMC estimator.

In trying to solve this problem, **we use coupling** (born to prove converge results of MCMC algorithms) which makes two chains “walk together” until they meet; the faster they meet, the faster convergence to the target distribution of the chain.
We can use this meeting time as an estimate on how quickly the chain forgets its starting point and use it to:
- **correct the bias from initialization** and 
- build an **unbiased estimator** without discarding samples.

---
## Formal Coupling
Let us consider a pair of MC $X_{t}, Y_{t}$ which share the same $\pi$-invariant kernel, $K$.
Then the joint transition kernel for the coupled chains $(X_{t},Y_{t})$ is given by: $$\bar{K}((x,y), \cdot)$$ s.t. $$\bar K((x,y), A \times \mathcal{X}) = K(x, A), \quad \bar K((x,y), \mathcal{X} \times A) = K(y, A)$$
and define the **meeting time** as $\tau := \inf\{t\geq0, X_{t}= Y_{t}\}$.

Few examples:
1. **Synchronous Coupling**: Use the same source of randomness for both chains. 
   Sample $U \sim Unif[0,1]$ then set $X' \sim \phi(X,U)$ and $Y'\sim\phi(Y,U)$
2. **Maximal Coupling:** Maximizes the meeting probability at each step. Useful when we need fast coupling $$\max_{X\sim p, Y\sim q}P(X=Y) = 1 - ||p-q||_{TV}$$
   where TV (total variation) actually measure how different are the densities in term of the probability they assign to same event. 

---
## Glynn–Rhee Unbiased Estimator

Given a target function $h$, define two coupled chains $(X_t, Y_t)$ as
$X_{0}\sim \pi_{0}$, $Y_{0}\sim \pi_{0}$ 
$X_{1}\sim K(X_0,\cdot)$ 
and set the coupled kernel as $$(X_{t+1},Y_{t})\sim \bar{K}((X_{t},Y_{t-1}),(\cdot,\cdot)), \quad \forall t\geq1$$
and meeting time $\tau$ as defined above.

Then the **Glynn–Rhee estimator** is:

$H = h(X_k) + \sum_{t = k+1}^{\tau-1} [h(X_t) - h(Y_{t-1})]$, for any fixed $k \ge 0$.

Under the assumptions:
1. $\mathbb{E}[h(X_{t})] \rightarrow \mathbb{E}_{\pi}[h(X)]$ for $t\rightarrow \infty$ 
2. meeting time decrease at geometric rate (i.e. $P(\tau >t)\leq C\delta^{t}$)
3. and that the chains stay together after they met (i.e. $X_t = Y_{t-1} \quad \forall t \geq \tau$)

we have: $\mathbb{E}[H] = \mathbb{E}_{\pi}[h(X)].$
### Proof Sketch of Unbiasedness
Using the telescoping expectation and Fubini's Theorem + A1:
$$\mathbb{E}_{\pi}[h(X)] = \lim_{t\to\infty} \mathbb{E}[h(X_t)] = \mathbb{E}\left[h(X_k) + \sum_{t=k+1}^{\infty} \left[h(X_t) - h(X_{t-1})\right]\right]$$

Replacing $h(X_{t-1})$ with $h(Y_{t-1})$ (by A3) (since the chains coincide after $\tau$) gives:

$\mathbb{E}\left[h(X_k) + \sum_{t=k+1}^{\tau-1} (h(X_t) - h(Y_{t-1}))\right] = \mathbb{E}[H]$, (by A2).

---
## **Convergence Diagnostic: a By-product**

**Idea**: The faster two coupled MCMC chains meet, the closer the chains are to stationarity. 
Now, we consider an extension of Glynn-Rhee whit a lag $L>1$ between the coupled chain we get that the meeting time $\tau = \inf\{t \geq L : X_t = Y_{t-L}\}$ acts as a proxy for convergence quality.

Now, if we simulate $\ell$ runs, by the coupling inequality we have: $$\|\pi_k - \pi\|_{TV} \leq E[\max(0,\frac{τ−L−k​}{L})].$$where:
- $\pi_k$ denotes the marginal distribution of the chain after $k$ iterations
- $k$ is the burn-in period,
- $L$ is a lag parameter.

The diagnostic replaces the expectation with an empirical estimate from $N$ independent coupling runs:
$$\frac{1}{N}\sum_{i=1}^N \max\left(0, \frac{\tau_i - L - k}{L}\right)$$
**Practical Usage**: For a desired tolerance $\epsilon > 0$, choose burn-in $\hat{k}_\epsilon$ as:
$$\hat{k}_\epsilon = \arg\min\left\{k : \frac{1}{N}\sum_{i=1}^N \max\left(0, \frac{\tau_i - L - k}{L}\right) < \epsilon\right\}$$
This tells you how many iterations are needed before your bias falls below $\epsilon$.

---
# Appendix
#### Maximal Coupling Algorithm
1. Sample $X \sim p$.
2. Draw $U \sim \text{Uniform}(0,1)$.
    - If $U \le \frac{q(X)}{p(X)}$, set $Y = X$ (the two chains meet).
    - Else, repeatedly sample $Y’ \sim q, W \sim \text{Uniform}(0,1)$
	      until $W > \frac{p(Y’)}{q(Y’)}$; set $Y = Y’.$
  