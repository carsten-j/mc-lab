## Motivation
HMC is a **gradient-informed method** that using Hamiltonian dynamics allows us to perform large non-local moves while maintaining high acceptance rates:

- Efficient state space exploration
- Better scaling with the dimension of the target $O(d^{-1/4})$ vs $O(d^{-1/3})$ for MALA vs $O(d^{-1})$ for RWM

---
## Physics Foundation

Consider a system with position $q \in \mathbb{R}^d$, momentum $p \in \mathbb{R}^d$, continuously differentiable potential function $U: \mathbb{R}^d \to \mathbb{R}$, and mass matrix $M \in \mathbb{R}^{d \times d}$ (symmetric positive definite).

**Hamiltonian energy:** $$H(q,p) = \underbrace{\frac{p^T M^{-1} p}{2}}_{K(p) \text{ (kinetic)}} + \underbrace{U(q)}_{\text{(potential)}}$$

**Key property:** For reasons that will become clear as we go, energy conservation — $H(q,p)$ remains constant along trajectories.

---

## Hamilton's Equations

The system evolves according to the following Hamilton's equations: 
Using $z = \begin{pmatrix} q \ p \end{pmatrix} \in \mathbb{R}^{2d}$ and canonical structure matrix $$J = \begin{bmatrix} 0 & I_d \ -I_d & 0 \end{bmatrix}$$we write compactly: $$\frac{d}{dt}z = J\nabla_z H(z)$$Where $J$ satisfies that $J^{-1} = J^T$.

---

## Hamiltonian Flow

By solving Hamiltonians equations the **Hamiltonian flow** $\Psi_{t,H}$ maps initial state $z(0)=(q(0),p(0))$ to $z(t)$ : $$\Psi_{t,H}(z(0)) = z(t)$$
### Key Properties

**1. Symplecticity:**  The Hamiltonian flow is a symplectic map. I.e. $[\nabla\Psi(z)]^T J^{-1} \nabla\Psi(z) = J^{-1}$

**2. Volume Preservation:** Symplecticity implies volume preservation: $|\det(\nabla \Psi)| = 1$

**3. Stationarity of $\pi(dz) \propto \exp(-H(z))dz$** with respect to the Hamiltonian flow.

**Proof:**

1. Let $Z_0 \sim \pi$ with density $\pi(z) \propto \exp(-H(z))$
2. Let $Z_t = \Psi_{t,H}(Z_0)$, hence $Z_{t}\sim \pi_{t}$
3. By change of variables: $$\pi_t(z_t) = \pi(z_0) \cdot |\det(\nabla \Psi)|^{-1} \cdot \frac{\exp(-H(z_0))}{\exp(-H(z_t))}$$
4. Using volume preservation ($|\det(\nabla \Psi)| = 1$) and energy conservation ($H(z_0) = H(z_t)$): $$\pi_t(z_t) = \pi(z_0) = \pi(z_t)$$
5. Therefore $Z_t \sim \pi$

### Ergodicity Issue

Hamilton's equations alone don't define an ergodic Markov chain — they preserve energy, preventing exploration across energy levels.

**Solution:** Alternate between:

1. **Momentum resampling:** $p \sim N(0, M)$ (breaks energy constraint)
2. **Hamiltonian flow:** $\Psi_{T,H}$ (explores within energy level)

The combination $P_R \circ \Psi_{T,H}$ preserves $\pi$ and is ergodic.

### Irreducibility

Under smoothness assumptions on $U$, the kernel $K$ combining momentum resampling, hamiltonian flow and random trajectory length sampling $T \sim \nu_T$ (positive density on $[0,\tau]$) is strongly $\pi$-irreducible, enabling global moves.

---

## Leapfrog Integration

Since exact Hamiltonian dynamics cannot be simulated, we use the **Leapfrog discretization scheme:**

$$p\left(t + \frac{\epsilon}{2}\right) = p(t) - \frac{\epsilon}{2}\nabla U(q(t))$$ $$q(t + \epsilon) = q(t) + \epsilon M^{-1} p\left(t + \frac{\epsilon}{2}\right)$$ $$p(t + \epsilon) = p\left(t + \frac{\epsilon}{2}\right) - \frac{\epsilon}{2}\nabla U(q(t + \epsilon))$$

**Advantages:**

- Tracks true Hamiltonian trajectory closely
- Numerically stable (does not diverge)
- **Symplectic** — preserves volume exactly despite discretization

**Notes:**
- (it can be shown that) Leapfrog map a symplectic map, hence is volume preserving.
- When $L=1$, HMC is equivalent to MALA (up to momentum flip).

---

## HMC Algorithm

HMC extends the state space from $q \in \mathbb{R}^d$ to $z = (q,p) \in \mathbb{R}^{2d}$, sampling from: $$\pi(z) \propto \exp(-H(z)) = \exp(-U(q)) \exp\left(-\frac{p^T M^{-1} p}{2}\right)$$

From this decomposition, $q$ and $p$ are independent, with marginal distributions:
- target distribution: $\pi(q) \propto \exp(-U(q))$,
- and $p \sim N(0, M)$.

### Algorithm Steps
- Let $\Psi_{\epsilon}$ denote the Leapfrog map of step-size $\epsilon$ and $\Psi_{\epsilon}^L$ the composition of $L$ leapfrog maps with $L\in \mathbb{N}$.
- Let $\bf{N}$ denote the map that negates the momentum (i.e. $\bf{N}(q,p) = (q, -p)$)
- Let's assume a Gaussian kernel with covariance matrix $\bf{M}$ for the momentum.

**HMC Iteration:**

1. **Resample momentum:** $p \sim N(0, M)$
    
2. **Propose via leapfrog:**
	    $$\begin{pmatrix} q^* \\ p^* \end{pmatrix} := N\left(\Psi_{\varepsilon}^{L} \begin{pmatrix} q \\ p \end{pmatrix}\right)$$
    1. Apply $L$ leapfrog steps 
    2. Flip momentum
3. **Accept/reject with probability:** $$\alpha = \min \{1, \exp(H(q,p) - H(q',p')) \}$$  $$= \min\{1, \exp\left(U(q) - U(q') + \frac{p^T M^{-1} p}{2} - \frac{(p')^T M^{-1} p'}{2}\right)\}$$
 
**Note:** Momentum flip ensures reversibility but is unnecessary in practice (momentum resampled next iteration).

### Proof: $\pi$-Invariance

**Theorem:** The target distribution $\pi(z) \propto \exp(-H(z))$ is invariant under the HMC Markov kernel, and $\pi$ is reversible with respect to the leapfrog-propose-accept step.

**Proof:**

1. **Step 1 preserves $\pi$:** Since $q$ and $p$ are sampled independently.
    
2. **Reversibility of Leapfrog-Propose-Accept Step:** Let $P_2$ denote the Markov kernel for step 2. A kernel $K$ is reversible with respect to $\pi$ if for all bounded measurable functions $f: \mathbb{R}^{2d} \times \mathbb{R}^{2d} \to \mathbb{R}$: $$\int \int f(x,y) \pi(dx) K(x,dy) = \int \int f(x,y) \pi(dy) K(y,dx)$$
    
3. **Structure of $P_2$:** The kernel $P_2(x, dy)$ is non-zero only for $y = \Psi(x)$ (accepted proposal) and $y = x$ (rejected proposal): $$\int \int f(x,y) \pi(dx) P_2(x,dy) = \int f(x, \Psi(x)) \min[1, e^{H(x) - H(\Psi(x))}] \pi(dx)$$ $$+ \int f(x,x) \left(1 - \min[1, e^{H(x) - H(\Psi(x))}]\right) \pi(dx)$$
    
4. **Change of Variables:** Let $y = \Psi(x)$. Then $x = \Psi(y)$ (since $\Psi$ is an involution: $\Psi(\Psi(z)) = z$ due to momentum flip). By volume preservation: $$\pi(dy) = \pi(dx) \cdot \frac{\exp(-H(y))}{\exp(-H(x))} = \pi(dx) \cdot e^{H(x) - H(y)}$$
    
5. **First Term Transformation:** $$\int f(x, \Psi(x)) \min[1, e^{H(x) - H(\Psi(x))}] \pi(dx)$$ $$= \int f(\Psi(y), y) \min[1, e^{H(\Psi(y)) - H(y)}] \pi(dx)$$ $$= \int f(\Psi(y), y) \min[1, e^{H(\Psi(y)) - H(y)}] e^{H(y) - H(\Psi(y))} \pi(dy)$$ $$= \int f(\Psi(y), y) \min[1, e^{H(y) - H(\Psi(y))}] \pi(dy)$$
    
6. **Second Term:** The rejected proposal term is symmetric by the same argument.
    
7. **Conclusion:** The reversibility condition holds, thus $\pi$ is invariant.
    

**Key Insight:** Energy conservation (approximately preserved by leapfrog) implies that $H(q,p) \approx H(q',p')$, leading to acceptance probabilities close to 1. This is why HMC achieves high acceptance rates while making large moves, a key advantage over Random Walk Metropolis.

### Why Momentum Flip Doesn't Matter

The kinetic energy $K(p) = \frac{p^T M^{-1} p}{2}$ is symmetric in $p$: $K(p) = K(-p)$. Therefore, flipping the momentum does not change the acceptance probability or the target distribution. The momentum flip is purely for ensuring reversibility in the theoretical analysis, but has no practical effect since momentum is resampled at each iteration anyway.

---

## Summary

**HMC** leverages Hamiltonian mechanics for efficient gradient-informed MCMC:

**Advantages:**

- Large non-local moves with high acceptance rates
- Better dimensional scaling: $O(d^{-1/4})$
- Natural adaptation to target geometry via energy conservation
- Symplectic structure ensures theoretical validity

**Requirements:**

- Gradient $\nabla U(q)$ must be available
- Higher per-iteration cost (but fewer iterations needed)
- Not geometrically ergodic for heavy/super-light tails
- Still require careful step-size ($\epsilon$) and leapfrog iteration number ($L$) tuning:
	- **Too Large:** If $\epsilon$ is too large, discretization error accumulates, causing low acceptance rate (due to poor energy conservation).
	- **Too Small:** If $\epsilon$ is too small the computational cost per iteration increases ($L$ must be larger for same distance) if $L$ is also small behaves like MALA (when $L=1$)


**Extensions:** **No-U-Turn Sampler (NUTS)**  (Hoffman & Gelman, 2014) automatically tunes step size and trajectory length via adaptive tree-building and U-turn detection, ensuring symmetry at each depth level and actually avoid the M-H rejection step.

---
--- 











# Appendix

## Performance Comparison
For 100D Gaussian:

|Method|Step Size Scaling|Mixing|Notes|
|---|---|---|---|
|RWM|$O(d^{-1/2})$|Slow, random walk|Local moves|
|MALA|$O(d^{-1/3})$|Better|Uses gradients|
|HMC|$O(d^{-1/4})$|Best|Hamiltonian trajectories|



## Proof 17: Symplecticity of Leapfrog

**Theorem:** Leapfrog is symplectic and volume-preserving.

**Proof sketch:**

1. **Composition:** If $\Psi_1, \Psi_2$ are symplectic, then $\Psi = \Psi_1 \circ \Psi_2$ is symplectic (chain rule + symplectic condition).
    
2. **Shear transformations:** Each leapfrog substep is a shear:
    
    - $p$-update: $\begin{pmatrix} q \ p - \epsilon \nabla U(q) \end{pmatrix}$ has Jacobian $\begin{pmatrix} I_d & 0 \ -\epsilon \nabla^2 U(q) & I_d \end{pmatrix}$
    - $q$-update: $\begin{pmatrix} q + \epsilon M^{-1}p \ p \end{pmatrix}$ has Jacobian $\begin{pmatrix} I_d & \epsilon M^{-1} \ 0 & I_d \end{pmatrix}$
3. **Verification:** We check the symplectic condition for the $p$-update: $$(\nabla \Psi(z))^T J^{-1} \nabla \Psi(z) = \begin{pmatrix} I_d & -\epsilon \nabla^2 U(q) \ 0 & I_d \end{pmatrix} \begin{pmatrix} 0 & -I_d \ I_d & 0 \end{pmatrix} \begin{pmatrix} I_d & 0 \ -\epsilon \nabla^2 U(q) & I_d \end{pmatrix}$$ $$= \begin{pmatrix} 0 & -I_d \ I_d & 0 \end{pmatrix} = J^{-1}$$
    
4. **Conclusion:** Leapfrog = composition of symplectic shears $\Rightarrow$ symplectic $\Rightarrow$ volume-preserving.
    

**Key insight:** Alternating $p$ and $q$ updates preserves geometric structure, ensuring stability and accuracy.