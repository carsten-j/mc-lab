# Slide sampling reversibility

Looking at page 4 of your slides about the "stepping out and shrinkage" procedure, I'll provide an informal proof of reversibility.

## Informal Proof of Reversibility

The key insight is that the slice sampler maintains detailed balance through its construction as a Gibbs sampler on the extended space (x, y).

### Setting up the Reversibility Condition

For reversibility, we need to show that the transition kernel T satisfies:
π(x)T(x → x') = π(x')T(x' → x)

### The Stepping Out and Shrinkage Procedure

The procedure works as follows:

1. Given current state x₀, sample y ~ Uniform[0, f(x₀)]
2. Create an initial interval of width w around x₀
3. **Step out**: Expand the interval by steps of size w until both endpoints have f(x) < y
4. **Shrinkage**: Sample x' uniformly from the expanded interval, and if f(x') < y, shrink the interval by cutting off the side containing x' (relative to x₀)

### Key Observations for Reversibility

**1. Deterministic Interval Construction**
Given x₀ and y, the stepping-out procedure deterministically creates the same maximal interval [L, R] that:

- Contains x₀
- Has f(L) < y and f(R) < y at the boundaries
- Contains all connected components of the slice {x: f(x) ≥ y} that include x₀

**2. Symmetry of the Procedure**
The crucial property is that if we start from any point x within the final interval and perform the same stepping-out procedure with the same y value, we get the *same* interval [L, R]. This is because:

- The stepping out moves by fixed increments w
- The procedure stops at the same boundaries regardless of starting position within the slice

**3. Uniform Sampling Within the Slice**
The shrinkage procedure ensures we sample uniformly from the slice S_y = {x: f(x) ≥ y} within the interval.

### Establishing Detailed Balance

For the transition x → x' to occur:

- We need y such that both x and x' are in the slice S_y
- The probability is proportional to (1/f(x)) × (1/|S_y ∩ [L,R]|)

For the reverse transition x' → x:

- With the same y value, x' would generate the same interval [L,R]
- The probability is proportional to (1/f(x')) × (1/|S_y ∩ [L,R]|)

Since the joint distribution satisfies:
p(x,y) ∝ 1_{0<y<f(x)}

We have π(x) ∝ f(x), and thus:

- Forward: π(x)T(x→x') ∝ f(x) × (1/f(x)) × (1/|S_y|) = 1/|S_y|
- Reverse: π(x')T(x'→x) ∝ f(x') × (1/f(x')) × (1/|S_y|) = 1/|S_y|

Therefore, detailed balance holds, establishing reversibility.

### Why the Shrinkage is Necessary

The shrinkage step is essential because it:

1. Maintains the uniform distribution over the slice
2. Ensures that rejected points properly update the sampling interval
3. Preserves the symmetry needed for reversibility - any point in the final accepted slice region could have been the starting point

Without proper shrinkage, the procedure would oversample regions near the initial point, breaking reversibility.
