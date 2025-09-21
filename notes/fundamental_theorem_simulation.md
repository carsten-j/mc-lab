# The Fundamental Theorem of Simulation

# Conversation transcript (Markdown with LaTeX)

## User

please tell me about the fundamental theorem of simulation

## Assistant

Most people use “fundamental theorem of simulation” to refer to the probability integral transform (also called the inverse CDF method). It underpins why a uniform random number generator is enough to simulate from essentially any distribution.

What it says

- Let $F$ be a cumulative distribution function (cdf). Define the generalized inverse

$$

F^{-1}(u) = \inf\{x : F(x) \ge u\}.

$$

- If $U \sim \mathrm{Uniform}(0,1)$, then $X = F^{-1}(U)$ has cdf $F$. Conversely, if $X$ has continuous cdf $F$, then $F(X) \sim \mathrm{Uniform}(0,1)$.

Why it’s fundamental

- It reduces random variate generation to: get $U \sim \mathrm{Uniform}(0,1)$ and apply a deterministic transform.

- Independence is preserved: independent uniforms transformed this way give independent samples from $F$.

- It generalizes: for vectors you can use conditional cdfs (Rosenblatt transform) or copulas to build multivariate distributions from uniforms.

- Many classic generators are special cases (e.g., exponential via inverse, some gammas via composition, Poisson via interarrival times).

Proof sketch (one line each direction)

- If $U \sim \mathrm{U}(0,1)$ and $X = F^{-1}(U)$, then for any $x$,

$$

\mathbb{P}(X \le x) = \mathbb{P}(U \le F(x)) = F(x).

$$

- If $F$ is continuous and $X \sim F$, then

$$

\mathbb{P}(F(X) \le u) = \mathbb{P}(X \le F^{-1}(u)) = u,

$$

so $F(X) \sim \mathrm{U}(0,1)$.

Examples

- Exponential($\lambda$): If $U \sim \mathrm{U}(0,1)$, then $X = -\ln(1-U)/\lambda$ (often just $-\ln U/\lambda$) is $\mathrm{Exp}(\lambda)$.

- Discrete distribution with probabilities $p_1,\dots,p_k$: Build cumulative sums $c_j = \sum_{i\le j} p_i$, then return $j$ such that $U \in (c_{j-1}, c_j]$.

- Poisson process($\lambda$): Generate i.i.d. $\mathrm{Exp}(\lambda)$ interarrival times by inverse transform and cumulate.

Practical notes

- $F^{-1}$ must be computable or well-approximated. When it isn’t, use alternatives: acceptance–rejection, composition/mixtures, alias method (for discrete), specialized methods (e.g., Ziggurat for normal), or numerical inversion.

- Handle endpoints carefully: avoid $U=0$ for transforms like $-\ln U$ (use $U \in (0,1]$).

- Quality of the uniform RNG matters (period, equidistribution, correlations). For variance reduction or quasi–Monte Carlo, replace PRNGs with low-discrepancy sequences and use inverse/other mappings.

If you had a different “fundamental theorem” in mind (e.g., in automata/verification or modeling-and-simulation theory), tell me the context and I can tailor the explanation.

---

## User

I think your answer is related to what I had in mind. Please see the image

## Assistant

Great—your screenshot is the “under-the-curve” version of the fundamental theorem of simulation. It’s a very simple but powerful idea:

Statement (informal)

- Let $f$ be a nonnegative, integrable function on the space of $x$ (think: an unnormalized density). Define the set

$$

A \;=\; \{(x,u) : 0 < u < f(x)\}.

$$

- If you draw a point $(X,U)$ uniformly from $A$, then the marginal distribution of $X$ has density $f(x)/\!\int f$.

- Conversely, if $X$ has density $f(x)/\!\int f$ and, given $X=x$, you draw $U$ uniformly on $(0,f(x))$, then $(X,U)$ is uniform on $A$.

- Therefore, simulating $X \sim f$ is equivalent to simulating $(X,U)$ uniformly under the graph of $f$ and discarding $U$.

Geometric intuition

- Picture the area under the curve $u=f(x)$. Throw a dart uniformly at random into this area. The $x$-coordinate of the dart follows the target density (up to normalization).

- This holds even if $f$ is only known up to a constant: $c=\int f$ might be unknown, but the uniform distribution on $A$ still exists conceptually, and algorithms can sample from it without knowing $c$.

Proof sketch

- The uniform density on $A$ is

$$

h(x,u) \;=\; \frac{1}{c}\,\mathbf{1}\{0<u<f(x)\}, \quad c=\int f(x)\,dx,

$$

the “area” of $A$.

- The marginal of $X$ is

$$

p_X(x) \;=\; \int h(x,u)\,du \;=\; \frac{1}{c}\int_0^{f(x)} du \;=\; \frac{f(x)}{c}.

$$

Why it matters (algorithms it powers)

- Acceptance–rejection (AR): You want to sample uniformly under $g(x)$ ($\propto$ target); instead sample uniformly under an easier envelope $M q(x)$, then keep only the points that fall in the subset under $g$. The kept points are uniform under $g$, so their $x$-marginal is the target.

- Algorithm: Draw $X \sim q$, $V \sim \mathrm{Uniform}(0,1)$. Accept $X$ if $V \le g(X)/(M q(X))$. Acceptance rate $= (\int g)/M$.

- Slice sampling: Do Gibbs sampling on the joint “under-the-curve” distribution.

- Given current $X$, draw $U \sim \mathrm{Uniform}(0,g(X))$.

- Then draw a new $X$ uniformly from the slice $S = \{x : g(x) > U\}$.

- The stationary joint is uniform on $A$, so the $X$-marginal is the target.

- Thinning for inhomogeneous Poisson processes: Uniform points under $\lambda(t)$ in the $(t,u)$-plane produce event times with rate $\lambda(t)$. Acceptance–rejection with envelope $\lambda_{\max}$ implements this.

Relationship to the probability integral transform (PIT)

- PIT says: if $U \sim \mathrm{Uniform}(0,1)$ and $X = F^{-1}(U)$, then $X$ has cdf $F$. It’s a 1D mapping from a uniform.

- The “under-the-curve” theorem is a geometric reformulation: sampling $X$ is equivalent to sampling uniformly under $f$ and taking the $x$-coordinate. In 1D, both views reduce sampling to uniform randomness; PIT uses horizontal slicing via $F$, while this theorem uses vertical slicing via $f$.

Practical notes

- Works in any dimension for $x$: $A \subset \mathbb{R}^{d+1}$.

- $f$ can be unnormalized; only pointwise evaluation up to a constant is needed by AR or slice sampling.

- Uniform-on-$A$ is rarely sampled directly; AR, slice, and related methods are ways to realize it.

- No boundedness is required beyond $\int f < \infty$ (the “area” must be finite).

Tiny example

- Target $f(x) = 2x$ on $(0,1)$ (Beta$(2,1)$); $c = \int_0^1 2x\,dx = 1$.

- $A = \{(x,u): 0<x<1,\; 0<u<2x\}$.

- Uniform on $A$ implies $p_X(x) = \int_0^{2x} du = 2x$, as desired.

- Acceptance–rejection with envelope $q(x)=1$ on $(0,1)$ and $M=2$: draw $X \sim \mathrm{Uniform}(0,1)$, $V \sim \mathrm{U}(0,1)$, accept if $V \le 2X/2 = X$. Accepted $X \sim \mathrm{Beta}(2,1)$.

---

## User

What is meant by " draw a point $(X, U)$ uniformly from $A$"

## Assistant

“Draw a point $(X, U)$ uniformly from $A$” means:

- $A = \{ (x, u) : 0 < u < f(x) \}$ is a subset of $\mathbb{R}^{d+1}$ with finite $(d+1)$-dimensional volume

$$

\mathrm{Vol}(A) = \int f(x)\,dx.

$$

- The uniform distribution on $A$ is the probability measure that assigns equal probability per unit volume on $A$. Formally, for any measurable $B \subseteq A$,

$$

\mathbb{P}\big((X,U)\in B\big) = \frac{\mathrm{Vol}(B)}{\mathrm{Vol}(A)}.

$$

- Equivalently, $(X, U)$ has joint density

$$

h(x,u) = \frac{1}{\mathrm{Vol}(A)} \,\mathbf{1}\{0<u<f(x)\},

$$

and $0$ outside $A$.

From this definition, the marginal of $X$ is

$$

p_X(x) \;=\; \int h(x,u)\,du \;=\; \frac{1}{\mathrm{Vol}(A)}\int_0^{f(x)} du \;=\; \frac{f(x)}{\int f}.

$$

How do you actually sample such a point?

- Direct rejection from a bounding box (when available):

1) Find a region $D$ containing the support of $x$ and a constant $M$ with $0 \le f(x) \le M$ for $x \in D$.

2) Sample $X$ uniformly from $D$ and $V$ uniformly from $(0,M)$.

3) If $V < f(X)$, accept and set $U = V$; otherwise, reject and repeat.

The accepted pair $(X,U)$ is uniform on $A$. The acceptance rate is

$$

\frac{\mathrm{Vol}(A)}{\mathrm{Vol}(D)\,M} \;=\; \frac{\int f}{\mathrm{Vol}(D)\,M}.

$$

- More general acceptance–rejection:

If you can’t get a simple $D$ and $M$, use an envelope $M q(x)$ with known sampler $q$ and $M \ge f(x)/q(x)$. Draw $X \sim q$ and $V \sim \mathrm{Uniform}(0,1)$; accept if

$$

V \le \frac{f(X)}{M q(X)}.

$$

The accepted $(X, U = V\cdot M q(X))$ is uniform on $A$.

- Slice sampling (Gibbs on the joint):

Given a current $x$, draw $U \sim \mathrm{Uniform}(0, f(x))$. Then draw a new $X$ uniformly from the slice

$$

S(U) = \{ x : f(x) > U \}.

$$

Iterating these steps produces a Markov chain whose stationary joint is uniform on $A$; the $X$-marginal is your target.

Key clarifications

- “Uniform on $A$” refers to uniformity with respect to volume in the $(x,u)$-space, not “$U \sim \mathrm{Uniform}(0,1)$” (here $U$ is just the second coordinate’s name).

- This uniform distribution exists whenever $\int f(x)\,dx < \infty$, even if the $x$-domain is unbounded or $f$ is unnormalized.

---

## User

Please add a bit more details to the "Proof sketch"

## Assistant

Here is a more detailed, but still compact, proof in both directions with the core ideas explicit.

Setup and notation

- Let $f: E \to [0,\infty)$ be measurable and integrable with

$$

c \;=\; \int_E f(x)\,dx \in (0,\infty).

$$

- Define

$$

A \;=\; \{ (x,u) \in E \times (0,\infty) : 0 < u < f(x) \}.

$$

Volume of $A$

- By Fubini,

$$

\mathrm{Vol}(A) \;=\; \int_E \int_0^{f(x)} du\,dx \;=\; \int_E f(x)\,dx \;=\; c.

$$

What “uniform on $A$” means

- The joint density of the uniform-on-$A$ distribution is

$$

h(x,u) \;=\; \frac{1}{c}\,\mathbf{1}\{0<u<f(x)\}.

$$

Direction 1: If $(X,U)$ is uniform on $A$, then $X$ has density $f/c$

- For any measurable $B \subseteq E$,

$$

\mathbb{P}(X \in B) \;=\; \int_B \int_0^{f(x)} \frac{1}{c}\,du\,dx

\;=\; \int_B \frac{f(x)}{c}\,dx.

$$

- Hence the marginal density of $X$ is $p_X(x) = f(x)/c$.

Direction 2: If $X$ has density $f/c$ and $U\,|\,X=x \sim \mathrm{Uniform}(0,f(x))$, then $(X,U)$ is uniform on $A$

- The conditional density is

$$

p_{U|X}(u|x) \;=\; \frac{1}{f(x)}\,\mathbf{1}\{0<u<f(x)\}.

$$

- Therefore the joint density is

$$

p_{X,U}(x,u) \;=\; p_X(x)\,p_{U|X}(u|x)

\;=\; \frac{f(x)}{c}\cdot \frac{1}{f(x)}\,\mathbf{1}\{0<u<f(x)\}

\;=\; \frac{1}{c}\,\mathbf{1}\{0<u<f(x)\}

\;=\; h(x,u).

$$

Equivalence

- The two directions show:

- Sampling $(X,U)$ uniformly from $A$ and discarding $U$ gives $X$ with density $f/c$.

- Conversely, sampling $X\sim f/c$ and then $U|X \sim \mathrm{Uniform}(0,f(X))$ yields a pair uniform on $A$.

Remarks

- No bounded support is required; only $c<\infty$.

- Works for $x \in \mathbb{R}^d$ by replacing lengths with volumes.

---

## User

Can you try again with fewer measure-theoretic details and more focus on intuition

## Assistant

Here’s an intuition-first proof, in 1D for simplicity; the same picture works in higher dimensions.

Setup

- Let $f(x) \ge 0$ be integrable. Let $A$ be the region under its graph:

$$

A \;=\; \{(x,u) : 0<u<f(x)\}.

$$

- Let $C$ be the total area under the curve:

$$

C \;=\; \int f(x)\,dx.

$$

- “Uniform on $A$” means: if you throw a dart that lands somewhere in $A$ with equal chance per unit area, then the chance of landing in any subregion $B \subseteq A$ is $\mathrm{Area}(B)/C$.

Direction 1: If $(X,U)$ is uniform on $A$, then $X$ has density $f(x)/C$

- Look at a very thin vertical strip between $x$ and $x+dx$. The part of $A$ inside this strip is a rectangle of height $f(x)$ and width $dx$, so its area is approximately $f(x)\,dx$.

- Because the dart is uniform over area, the probability that it lands in that strip (i.e., that $X \in [x,x+dx]$) is about $f(x)\,dx / C$.

- As $dx \to 0$, that “probability per unit width” is the density of $X$. Hence the marginal density of $X$ is $f(x)/C$.

Direction 2: If $X$ has density $f(x)/C$ and $U \mid X=x$ is $\mathrm{Uniform}(0,f(x))$, then $(X,U)$ is uniform on $A$

- Given $X=x$, drawing $U$ uniformly on $(0,f(x))$ spreads probability evenly along the vertical segment above $x$.

- Consider a tiny rectangle $[x,x+dx]\times [u,u+du]$ inside $A$. The chance that $X \in [x,x+dx]$ is about $(f(x)/C)\,dx$; conditional on that, the chance that $U \in [u,u+du]$ is about $du/f(x)$.

- Multiply: $(f(x)/C)\,dx \times (du/f(x)) = (1/C)\,dx\,du$. This is the same constant for every tiny rectangle inside $A$, i.e., probability is proportional to area everywhere in $A$.

- Therefore the joint distribution is uniform over $A$.

Takeaways

- The $x$-locations where $f(x)$ is larger correspond to taller vertical “columns” under the curve, hence more area, hence higher probability in exact proportion to $f(x)$.

- You don’t need to know $C$ to use this idea in algorithms; $C$ cancels out. That’s why acceptance–rejection and slice sampling can work with unnormalized $f$.

- In higher dimensions, replace the thin “strip” by a thin column above a small $d$-dimensional patch around $x$; its “volume” is $f(x)$ times the patch size, so the same logic gives the $X$-marginal proportional to $f(x)$.