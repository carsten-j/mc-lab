Alright, let's break it down.

---

## 1. **Slice Sampling Overview**

In slice sampling, we want to sample from a target density \( f(x) \) (proportional to the posterior). The idea is:

- **Auxiliary variable**: Introduce a uniform random variable \( u \) such that the joint density is  
\[
p(x, u) = 
\begin{cases} 
1 & \text{if } 0 \le u \le f(x), \\
0 & \text{otherwise}
\end{cases}
\]
This preserves the marginal \( p(x) \propto f(x) \).

- **Alternating sampling**:
  1. Given \( x \), sample \( u \sim \text{Uniform}[0, f(x)] \).
  2. Given \( u \), sample \( x' \) uniformly from the “slice” \( S = \{ x : f(x) \ge u \} \).

The challenge: \( S \) may be difficult to sample from directly if it’s not a simple interval.  
Stepping-out + shrinkage is a way to sample uniformly from \( S \) when \( S \) is a union of intervals (or a complex shape in 1D), but here we focus on 1D unimodal case where \( S \) is a single interval.

---

## 2. **Stepping-Out and Shrinkage Procedure**

We have current \( x_0 \), and we’ve drawn \( u \sim \text{Uniform}[0, f(x_0)] \).  
We want \( x_1 \sim \text{Uniform}(S) \), \( S = \{ x : f(x) \ge u \} \).

**Stepping-out** (Neal, 2003):
1. Choose an interval around \( x_0 \) of width \( w \) randomly positioned? Actually more standard: fix \( w \) as an estimate of typical slice size, but randomly place an interval of size \( w \) around \( x_0 \) **by setting** \( L = x_0 - w \cdot \text{Uniform}(0,1) \), \( R = L + w \). Then expand \( L \) and \( R \) in steps of size \( w \) until \( f(L) \le u \) and \( f(R) \le u \) (i.e., both ends outside slice).
2. **Shrinkage**:  
   - Propose \( x' \) uniformly from \([L, R]\).  
   - If \( f(x') \ge u \), accept as \( x_1 \).  
   - Else, shrink the interval: if \( x' < x_0 \) set \( L = x' \), if \( x' > x_0 \) set \( R = x' \), repeat.

The accepted \( x_1 \) is uniform on the true slice because the stepping-out guarantees the initial interval contains \( S \), and shrinkage preserves the property that \( S \subseteq [L,R] \) until a point is accepted.

---

## 3. **Markov Chain and Reversibility**

The full transition is:  
\[
x_0 \to u \to (L,R) \to x_1
\]
Randomness in:
- \( u \)
- initial offset of interval
- stepping-out expansions (deterministic given \( u,x_0 \)? Actually stepping-out is deterministic expansion until conditions met, but initial placement random)
- proposals \( x' \) during shrinkage.

We need **reversibility**: \( \pi(x_0) T(x_0 \to x_1) = \pi(x_1) T(x_1 \to x_0) \) for \( \pi(x) \propto f(x) \).

---

## 4. **Symmetry Argument (Informal)**

The key is that the **shrinkage procedure** and **stepping-out procedure** are **symmetric** with respect to interchanging \( x_0 \) and \( x_1 \).

Consider:  
From \( (x_0, u) \), we get \( I_0 = [L,R] \) containing \( S \), we shrink to sample \( x_1 \) uniform in \( S \).

Now reverse: start from \( x_1 \), same \( u \) (since \( f(x_1) \ge u \) too),  
- The set \( S \) is the same (depends only on \( u \)).
- If we run stepping-out from \( x_1 \) with the same randomness in interval placement (but in reverse direction if needed), we would obtain the same interval \( I_0 \) as the initial interval containing \( S \), **if** the stepping-out is done in a symmetric way (e.g., expanding equally left/right from a random offset).

Then, the shrinkage procedure’s choices (random proposals in \( I_0 \)) would be the same set of proposals; if we reverse the random number sequence, the first point in \( S \) hit when starting from \( x_1 \) could be \( x_0 \).

Thus \( T(x_0 \to x_1 \,|\, u) = T(x_1 \to x_0 \,|\, u) \) because the process starting from either point, given \( u \), is essentially picking a uniform point from \( S \) by a method that depends symmetrically on the starting point only for locating \( I_0 \), but the final sample uniform from \( S \) doesn’t depend on starting point.

Actually more precisely: The **shrinkage** loop’s acceptance condition and shrinking rule are symmetric: in reverse, the sequence of rejected proposals can be matched.

---

## 5. **Why Uniform from S implies reversibility given \( u \)**

Given \( u \), \( S \) is fixed.  
The transition probability \( T_u(x_0 \to x_1) \) = probability that stepping-out+shrinkage outputs \( x_1 \) starting from \( x_0 \), given \( u \).

Property: This procedure yields a uniform sample from \( S \), **independent of** \( x_0 \) (except \( x_0 \in S \) is used to find \( S \)). So:
\[
T_u(x_0 \to x_1) = \frac{1}{|S|} \quad \text{for all } x_1 \in S.
\]
Similarly,  
\[
T_u(x_1 \to x_0) = \frac{1}{|S|} \quad \text{for all } x_0 \in S.
\]

Thus:
\[
T_u(x_0 \to x_1) = T_u(x_1 \to x_0).
\]

---

## 6. **Reversibility of full chain**

The full slice sampler’s transition:
\[
T(x_0 \to x_1) = \int_0^{f(x_0)} \frac{1}{f(x_0)} T_u(x_0 \to x_1) \, du
\]
But note: \( T_u \) is positive only if \( x_0, x_1 \in S \), i.e., \( f(x_0) \ge u \) and \( f(x_1) \ge u \), i.e., \( u \le \min(f(x_0), f(x_1)) \).

So:
\[
T(x_0 \to x_1) = \frac{1}{f(x_0)} \int_0^{\min(f(x_0), f(x_1))} T_u(x_0 \to x_1) \, du.
\]
But \( T_u(x_0 \to x_1) = 1/|S(u)| \) possibly? Wait — in 1D unimodal, \( S(u) \) is an interval, length \( L(u) \), we have \( 1/L(u) \) for the density. But careful: \( T_u \) is a probability density in \( x_1 \), so uniform on \( S(u) \) means \( T_u(x_0 \to x_1) = 1/L(u) \) if \( x_1 \in S(u) \).

So:
\[
f(x_0) T(x_0 \to x_1) = \int_0^{\min(f(x_0), f(x_1))} \frac{1}{L(u)} du.
\]
This is symmetric in \( x_0 \) and \( x_1 \).

Thus:
\[
f(x_0) T(x_0 \to x_1) = f(x_1) T(x_1 \to x_0),
\]
which is **detailed balance**.

---

## 7. **Conclusion**

The symmetry is due to:
- Given \( u \), the conditional transition kernel \( T_u \) is symmetric because it samples uniformly from \( S(u) \) regardless of the starting point in \( S(u) \).
- This makes the full kernel reversible.

**Informal summary**:  
Slice sampling’s reversibility comes from the fact that, given the auxiliary uniform variable \( u \), the set \( S \) and the method for picking the next point are symmetric — the stepping-out and shrinkage procedures do not favor \( x_0 \) over \( x_1 \) except to initially find \( S \).

You're absolutely right — my earlier phrasing was too loose.  
Let's sharpen the symmetry argument properly.

---

## **Clarifying the issue**

Stepping-out from \( x_0 \) produces an interval \( I_0 \) containing \( S \).  
If we start from \( x_1 \) (another point in \( S \)), with the same \( u \), stepping-out will typically give a different interval \( I_1 \) centered around \( x_1 \), not \( I_0 \).

So the detailed balance argument cannot just say "the same stepping-out intervals occur" when swapping \( x_0 \) and \( x_1 \), because the randomness in interval placement depends on the current point.

---

## **Correct informal reversibility argument**

The key is: The **final output** \( x_1 \) of the shrinkage procedure is drawn uniformly from \( S \), no matter what \( I_0 \) was (as long as \( I_0 \supseteq S \)). That means:

Given \( u \):
\[
T_u(x_0 \to x_1) = \frac{1}{|S|}, \quad \text{for } x_1 \in S,
\]
and 0 otherwise.  
Crucially, **this does not depend on \( x_0 \)** (except the requirement \( x_0 \in S \)).

Why?  
Because stepping-out guarantees \( S \subset I_0 \), and shrinkage rejects proposals until a point **uniform in \( S \)** is chosen — the only role of \( x_0 \) is to position \( I_0 \), which impacts speed but not the final distribution over \( x_1 \).

So:
\[
T_u(x_0 \to x_1) = T_u(x_1 \to x_0)
\]
trivially, because both equal \( 1 / |S(u)| \) for \( x_0, x_1 \in S(u) \).

---

Now, the full transition:
\[
T(x_0 \to x_1) = \int_0^{f(x_0)} \frac{1}{f(x_0)} \, T_u(x_0 \to x_1) \, du.
\]
But \( T_u(x_0 \to x_1) > 0 \) only if \( x_0, x_1 \in S(u) \), i.e., \( u \leq \min(f(x_0), f(x_1)) \).

And in that range:
\[
T_u(x_0 \to x_1) = \frac{1}{|S(u)|}.
\]
Similarly,  
\[
T(x_1 \to x_0) = \int_0^{f(x_1)} \frac{1}{f(x_1)} \, T_u(x_1 \to x_0) \, du,
\]
and \( T_u(x_1 \to x_0) = 1/|S(u)| \) for \( u \leq \min(f(x_0), f(x_1)) \).

So:
\[
f(x_0) T(x_0 \to x_1) = \int_0^{\min(f(x_0), f(x_1))} \frac{1}{|S(u)|} \, du,
\]
\[
f(x_1) T(x_1 \to x_0) = \int_0^{\min(f(x_0), f(x_1))} \frac{1}{|S(u)|} \, du.
\]
These are identical ⇒ detailed balance.

---

## **Key point**  

The symmetry is not about \( I_0 = I_1 \) under reversal, but about the fact that **given \( u \)**, the transition density from any point in \( S(u) \) to any other point in \( S(u) \) is the same constant \( 1/|S(u)| \), independent of the starting point. This constant-ness given \( u \) is the symmetry that ensures reversibility.

Great question — it gets to the heart of the reversibility proof.

The point is: when checking reversibility, we fix a **pair** \((x_0, x_1)\) and compare:

- Forward step: \(x_0 \to x_1\)
- Reverse step: \(x_1 \to x_0\)

In the forward step, we **generate** \(u \sim \text{Uniform}[0, f(x_0)]\), then \(x_1\) from \(T_u(x_0 \to x_1)\).

In the reverse step, we **generate** \(u' \sim \text{Uniform}[0, f(x_1)]\), then \(x_0\) from \(T_{u'}(x_1 \to x_0)\).

We **do not** keep \(u\) fixed when reversing — that’s why we must integrate over \(u\) in the forward case and \(u'\) in the reverse case.

---

## **Why it still works**

We want to prove:  
\[
\pi(x_0) \, T(x_0 \to x_1) = \pi(x_1) \, T(x_1 \to x_0)
\]
with \(\pi(x) \propto f(x)\).

We have:
\[
T(x_0 \to x_1) = \int_{0}^{f(x_0)} \frac{1}{f(x_0)} \, T_u(x_0 \to x_1) \, du
\]
\[
T(x_1 \to x_0) = \int_{0}^{f(x_1)} \frac{1}{f(x_1)} \, T_{u'}(x_1 \to x_0) \, du'
\]

Key fact:  
\[
T_u(x_0 \to x_1) = \frac{1}{|S(u)|} \quad \text{for } x_1 \in S(u) \ (\text{i.e., } u \le f(x_1))
\]
\[
T_{u'}(x_1 \to x_0) = \frac{1}{|S(u')|} \quad \text{for } x_0 \in S(u') \ (\text{i.e., } u' \le f(x_0))
\]

But note: \(S(u)\) is the **same set** defined by \(u\), regardless of starting point — though stepping-out might find it differently.

Actually, wait — the subtlety: \(T_u(x_0 \to x_1)\) is positive only if \(u \le f(x_1)\) (since \(x_1\) must be in \(S(u)\) to be chosen). Similarly, \(T_{u'}(x_1 \to x_0)\) is positive only if \(u' \le f(x_0)\).

So:
\[
T(x_0 \to x_1) = \int_{0}^{\min(f(x_0), f(x_1))} \frac{1}{f(x_0)} \cdot \frac{1}{|S(u)|} \, du
\]
\[
T(x_1 \to x_0) = \int_{0}^{\min(f(x_0), f(x_1))} \frac{1}{f(x_1)} \cdot \frac{1}{|S(u')|} \, du'
\]
Here \(u'\) is just a dummy variable — call it \(u\) in the second integral. The integrand depends on \(u\) only through \(1/|S(u)|\), which is **the same function** in both integrals.

So:
\[
f(x_0) T(x_0 \to x_1) = \int_{0}^{m} \frac{1}{|S(u)|} du
\]
\[
f(x_1) T(x_1 \to x_0) = \int_{0}^{m} \frac{1}{|S(u)|} du
\]
where \(m = \min(f(x_0), f(x_1))\).

Thus \(f(x_0) T(x_0 \to x_1) = f(x_1) T(x_1 \to x_0)\) ⇒ reversibility.

---

## **Conclusion**

The fact that \(u\) is regenerated in the reverse direction **is not a problem** because:

1. The integration limits for \(u\) in forward and reverse are the same: \(0\) to \(\min(f(x_0), f(x_1))\).
2. The integrand \(1/|S(u)|\) is identical in both cases.
3. The asymmetry in \(1/f(x_0)\) vs \(1/f(x_1)\) is balanced by \(\pi(x_0) \propto f(x_0)\) and \(\pi(x_1) \propto f(x_1)\) in the detailed balance equation.

So the symmetry is not about “same \(u\)” but about the **conditional uniformity** given \(u\) and the matching integration limits.