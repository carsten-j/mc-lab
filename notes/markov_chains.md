# Summary: Markov chains and MCMC (key properties and implications)

A discrete-time process $(X_t)_{t \ge 1}$ is called a Markov chain if is satisfies the Markov property: $$\Pr(X_{n+1}=x_{n+1} \mid X_n=x_{n},\ldots,X_1=x_1) = \Pr(X_{n+1}=x_{n+1} \mid X_n=x_{n})$$
i.e, the future is conditionally independent of the past given the present.

The chain is called time-homogeneous if:

$$\forall m \in N : \Pr(X_{t}=y \mid X_{t-1}= x) = \Pr(X_{t+m} = y \mid  X_{t+m-1} = x )$$
i.e, the transition probability of moving from state $x$ to $y$ is independent of time $t$.

- Homogeneous Markov chain

     Discrete-time chain whose transition probabilities do not depend on time:
        - Markov property:
            -  $\Pr(X_{n+1}=j \mid X_n=i,\ldots,X_0) = \Pr(X_{n+1}=j \mid X_n=i)$.
        - Time-homogeneous:
            - $\Pr(X_{n+1}=j \mid X_n=i) = p_{ij}$ (independent of (n)$.
        - Evolution: $\mu_{n+1} = \mu_n P$, so  $\mu_n = \mu_0 P^n$.
- MCMC key idea and needed properties

  - Build a chain with target $\pi$ such that $X_t \Rightarrow \pi$ and
    - $\frac{1}{n}\sum_{t=1}^n \phi(X_t) \to \int \phi , d\pi$ (a.s.).
  - Sufficient qualitative conditions (discrete case): irreducible + positive recurrent + aperiodic.
  - General-state-space analogue: Harris ergodic.
  - “Nice to have” for efficiency: geometric ergodicity/spectral gap → CLT and finite asymptotic variance.
- Single stronger conditions implying irreducible + aperiodic + positive recurrent

  - Finite state space: primitive (regular) matrix:
    - $\exists m$ with $P^m > 0$ (all entries strictly positive).
  - General state space: Doeblin/minorization:
    - $\exists n_0,\varepsilon>0,\nu$ s.t. $P^{n_0}(x,A) \ge \varepsilon \nu(A)$ for all (x,A).
- Detailed balance (DB): role and limits

  - DB: $\pi(dx)P(x,dy) = \pi(dy)P(y,dx) \Rightarrow \pi P = \pi$ (stationarity).
  - DB alone does not imply irreducibility, aperiodicity, or positive recurrence.
  - Counterexamples: reducible block-diagonal reversible chains; 2-state flip (period 2) reversible w.r.t. $\pi=(1/2,1/2)$.
- Aperiodicity via self-transitions

  - If $P(i,i)>0$ for all (i), then the chain is aperiodic.
  - In an irreducible chain, having $P(k,k)>0$ for a single state (k) suffices (common period becomes 1).
  - Trick for general spaces: “lazy” kernel $\tilde P=(1-\varepsilon)I+\varepsilon P$ preserves $\pi$ and guarantees aperiodicity.
- Invariant (stationary) distribution

  - Discrete: $\pi P=\pi$ i.e. $\pi_j = \sum_i \pi_i P_{ij}$.
  - General: $\pi(A) = \int P(x,A),\pi(dx)$ for all measurable (A).
  - Equivalently: if $X_0\sim\pi$, then $X_t\sim\pi$ for all (t).
- Irreducibility: useful sufficient conditions

  - Discrete/finite:
    - Strong connectivity of the directed graph with edges $i\to j$ when $P_{ij}>0$.
    - Primitive matrix (as above).
    - Mixtures/compositions preserve irreducibility: $wP_1+(1-w)P_2$, $P_2P_1$.
  - General-state-space:
    - Doeblin/minorization (uniform) ⇒ $\varphi$-irreducible (and more).
    - Small set (C) with minorization plus reachability of (C) from any (x).
  - MCMC-specific:
    - Metropolis–Hastings: if proposal density $q(x,y)>0$ wherever $\pi(y)>0$, the MH kernel is irreducible on $\mathrm{supp}(\pi)$.
    - Random-walk MH: if $q(z)>0$ in a neighborhood of (0) and $\mathrm{supp}(\pi)$ is path-connected with nonempty interior, irreducible on that support.
    - Independent MH: proposal $g(y)>0$ on $\mathrm{supp}(\pi)$ ⇒ irreducible.
    - Gibbs: positivity of the joint on a connected support and positive full-conditionals on their supports.
    - Adding a small-probability global move with full support (mixture proposal) typically ensures irreducibility.
- Practical checklist for MCMC

  - Ensure $\pi$ is invariant (global balance; DB is a convenient sufficient condition).
  - Ensure ergodicity: irreducible, aperiodic, and (positive/Harris) recurrent.
  - For robustness and speed, prefer designs with geometric ergodicity when possible.

- Time-homogeneous means the transition kernel P does not depend on time. That alone does not make the chain stationary.

What “stationary” requires

- Existence of an invariant distribution π satisfying πP = π.
- The process must be started in that distribution: if X0 ~ π, then Xt ~ π for all t.

Why homogeneity is not enough

- A time-homogeneous chain may have no invariant probability distribution at all (e.g., simple symmetric random walk on the integers).
- Even when an invariant distribution exists, the chain is not a stationary process unless the initial law equals π. Example: the 2-state flip P = [[0,1],[1,0]] has π = (1/2, 1/2), but starting from δ0 the marginals alternate and are not constant over time.

Nuances

- Finite state space: any time-homogeneous chain has at least one stationary distribution. Uniqueness and convergence to it require irreducibility and aperiodicity (then positive recurrence holds automatically).
- Infinite/general state spaces: homogeneity does not guarantee existence. Positive (Harris) recurrence is typically needed for an invariant probability distribution; detailed balance or global balance is a common way to verify πP = π once π exists.

In other words, the detailed balance equations (1.38) hold, and hence the Markov chain is time reversible and has stationary probabilities { π i } . Moreover, this stationary distribution is also the limiting distribution if the Markov chain is irreducible and aperiodic.

 detailed balance implies stationary - se MCMC slides 

So far we did not touch upon terms like Transience and recurrence. What do they mean and how are the related to properties that we already talked about

Great question. Here’s a concise overview of transience and recurrence and how they connect to the properties we discussed.

Core definitions (discrete state space)

- Hitting/return times:
  - τ_i^+ = first return time to state i after time 0.
- Recurrent vs transient:
  - State i is recurrent if P_i(τ_i^+ < ∞) = 1 (you return to i eventually with probability 1); otherwise it’s transient.
  - Equivalent test: i is recurrent iff Σ_{n=0}^∞ P^n(i,i) = ∞; transient iff that sum is finite.
- Positive vs null recurrence:
  - Positive recurrent: E_i[τ_i^+] < ∞.
  - Null recurrent: recurrent but E_i[τ_i^+] = ∞.

Class-level facts

- In an irreducible chain, all states share the same type: either all transient, all null recurrent, or all positive recurrent.
- The period is also common to all states in an irreducible chain (periodicity is independent of recurrence).

Links to invariant/stationary distributions

- Existence/uniqueness:
  - If an irreducible chain is positive recurrent, it has a unique invariant probability distribution π.
  - If an irreducible chain has an invariant probability distribution, then it must be positive recurrent.
- Kac’s formula (discrete): for positive recurrent chains, π_i = 1 / E_i[τ_i^+]. More generally, for a set A with π(A) > 0, the expected return time to A starting from stationarity is 1/π(A).

Relation to aperiodicity and convergence

- Aperiodicity is orthogonal to (positive) recurrence. A chain can be positive recurrent yet periodic.
- Distributional convergence: irreducible + positive recurrent + aperiodic ⇒ the law of X_t converges to π from any start.
- Ergodic averages: for irreducible positive recurrent chains, time averages of integrable functions typically converge to E_π[φ] (periodicity can prevent marginal convergence but not the law of large numbers for averages).

General-state-space analogue (MCMC context)

- Harris recurrence: the chain visits every set with positive reference measure infinitely often (almost surely).
- Positive Harris recurrence: Harris recurrent and admits an invariant probability π; this is the right notion for MCMC.
- With aperiodicity, positive Harris recurrence implies:
  - Xt → π in distribution (and often in total variation),
  - strong laws for ergodic averages (the MCMC “key idea”).

How this ties back to earlier properties

- Irreducibility + positive recurrence are the backbone for having a unique invariant π and valid ergodic averages.
- Aperiodicity ensures actual convergence of the marginal distribution to π.
- Detailed balance only ensures invariance (πP = π); it does not imply irreducibility or (positive) recurrence.
- Stronger conditions (e.g., Doeblin/minorization; primitive matrices in finite spaces) imply irreducible + aperiodic + positive (Harris) recurrence.

Examples

- Simple symmetric random walk on Z: recurrent but null recurrent (E[return time] = ∞), so no invariant probability distribution.
- Random walk on Z^d: null recurrent for d = 1,2; transient for d ≥ 3.
- Two-state flip (0 ↔ 1): positive recurrent and periodic (period 2); π = (1/2, 1/2); time averages converge but the marginal distribution oscillates.
- Any finite irreducible chain: automatically positive recurrent; with aperiodicity you get convergence to the unique π.

I am not sure that I understand the difference between recurrent and positive recurrent

Here’s the clean distinction.

- Recurrent (for a state i): starting from i, you return to i with probability 1. Formally, if τ_i^+ is the first return time to i after time 0, then P_i(τ_i^+ < ∞) = 1. Returns happen infinitely often a.s., but they might be very rare.

- Positive recurrent: recurrent and the expected return time is finite, E_i[τ_i^+] < ∞. So you come back not only surely but also “quickly on average.”

- Null recurrent: recurrent but E_i[τ_i^+] = ∞. You will return eventually with probability 1, but the average wait is infinite.

Key relationships

- In an irreducible chain, all states share the same classification (all transient, all null recurrent, or all positive recurrent).

- Existence of stationary distribution:

  - Irreducible + positive recurrent ⇔ there exists a unique invariant probability distribution π.
  - Kac’s formula (discrete): for positive recurrent chains, π_i = 1 / E_i[τ_i^+].
  - Null recurrent chains have no invariant probability distribution (the natural invariant measure is infinite, not normalizable).
- Aperiodicity is separate: a chain can be positive recurrent yet periodic (then marginals may oscillate, though time averages still converge).

Examples

- Simple symmetric random walk on Z: recurrent but null recurrent (returns a.s., but E[return time] = ∞) → no stationary distribution.

- Any finite irreducible Markov chain: automatically positive recurrent (finite expected return times), hence has a unique stationary distribution.

- Birth–death chains with a strong drift toward a bounded region (e.g., reflecting boundaries or inward drift) are typically positive recurrent; with no confining drift they can be null recurrent or transient.

Do we use the Homogeneous property at all to ensure the MCMC worksi, ie. MCMC converges to target dist and integral is well-behaved

Short answer: Not by itself. Time-homogeneous (same transition kernel at every step) is standard in MCMC, but it neither guarantees convergence to the target nor is it strictly necessary.

How MCMC “works” (homogeneous case)

- What you actually need:
  - Invariance: pick a kernel P with πP = π (often via detailed balance).
  - Ergodicity: irreducible + positive (Harris) recurrence; add aperiodicity if you want Xt ⇒ π (marginal convergence). For the LLN of averages, positive Harris recurrence suffices.
- Then the Markov chain ergodic theorem gives, for integrable φ, 1/n ∑t φ(Xt) → Eπ[φ] almost surely.

What homogeneity buys you

- A single, time-invariant kernel P makes it easy to:
  - Verify π is invariant (πP = π).
  - Use standard ergodic theorems, spectral-gap/mixing-time tools, drift–minorization, etc.
- But homogeneity alone is far from enough (e.g., a homogeneous periodic chain need not converge in distribution).

Is homogeneity necessary?

- No. There are useful non-homogeneous/adaptive algorithms.
  - If you change kernels over time {Pt}, you can still preserve π at each step (πPt = π), but convergence from arbitrary starts is not automatic.
  - To get guarantees you need extra conditions (typical in adaptive MCMC): diminishing adaptation (Pt+1 ≈ Pt in total variation, uniformly) and containment (mixing times don’t blow up). Without these, the chain can fail to converge even if every Pt preserves π.

Practical takeaway

- In the sampling phase, most MCMC is deliberately time-homogeneous so we can apply standard results: prove πP = π and check ergodicity (irreducible, positive Harris recurrent; often aperiodic via self-transitions).
- Warm-up/adaptation is usually non-homogeneous; we discard it or ensure the extra conditions above if we want theory during adaptation.
- So: homogeneity is convenient and commonly assumed, but the properties that ensure MCMC “works” are invariance plus ergodicity (and, for marginal convergence, aperiodicity).
