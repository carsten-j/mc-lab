# Hamiltonian Monte Carlo (HMC) Implementation Plan

## Overview

Create a simple, educational implementation of Hamiltonian Monte Carlo in `src/mc_lab/hmc.py` that follows the existing codebase patterns and uses PyTorch for automatic gradient computation.

## Core Components

### 1. Class-Based Interface

**Class: `HMCSampler`**
- Follows the pattern established in `MALA_auto_grad.py`
- Provides a clean, reusable interface for HMC sampling
- Encapsulates all HMC-specific parameters and logic

### 2. PyTorch Integration

**Automatic Differentiation:**
- Use `torch.autograd.grad()` for computing gradients of log-target density
- Eliminates need for users to provide analytical gradients
- Handles conversion between NumPy arrays and PyTorch tensors

### 3. Leapfrog Integrator

**Standard Discretization:**
- Implements the leapfrog method for Hamiltonian dynamics
- Half-step momentum updates
- Full-step position updates
- Symmetric and volume-preserving

**Algorithm:**
```
p = p - (ε/2) * ∇U(q)      # half step for momentum
for i in range(L-1):
    q = q + ε * p           # full step for position
    p = p - ε * ∇U(q)       # full step for momentum
q = q + ε * p               # full step for position
p = p - (ε/2) * ∇U(q)      # half step for momentum
```

where U(q) = -log π(q) is the potential energy.

### 4. ArviZ Integration

**Returns `az.InferenceData`:**
- Uses existing `create_inference_data()` helper from `_inference_data.py`
- Provides comprehensive MCMC diagnostics
- Stores posterior samples and sample statistics
- Includes acceptance rates, log-likelihood values

### 5. Standard RNG Interface

**Consistent with Codebase:**
- Use `RandomState` type from `_rng.py`
- Use `as_generator()` for normalization
- Ensures reproducibility with seeds

## Class Design

### Constructor Parameters

```python
class HMCSampler:
    def __init__(
        self,
        log_target: Callable[[torch.Tensor], torch.Tensor],
        step_size: float = 0.1,
        n_leapfrog_steps: int = 10,
        var_names: Optional[List[str]] = None,
    )
```

**Parameters:**
- `log_target`: Function computing log π(x), accepts PyTorch tensors, returns scalar
- `step_size` (ε): Step size for leapfrog integrator (default: 0.1)
- `n_leapfrog_steps` (L): Number of leapfrog steps per proposal (default: 10)
- `var_names`: Optional list of variable names for ArviZ output

### Sample Method

```python
def sample(
    self,
    n_samples: int = 1000,
    n_chains: int = 4,
    burn_in: int = 1000,
    thin: int = 1,
    initial_states: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
    progressbar: bool = True,
) -> az.InferenceData
```

**Parameters:**
- `n_samples`: Number of samples per chain (after burn-in and thinning)
- `n_chains`: Number of independent chains
- `burn_in`: Number of initial samples to discard
- `thin`: Keep every thin-th sample
- `initial_states`: Optional initial states for chains
- `random_seed`: Seed for reproducibility
- `progressbar`: Show progress bar during sampling

**Returns:**
- `az.InferenceData` with posterior samples and diagnostics

## Implementation Details

### Helper Methods

1. **`_compute_gradient(x: torch.Tensor) -> torch.Tensor`**
   - Compute gradient using PyTorch autograd
   - Handle detach/requires_grad properly

2. **`_log_target_numpy(x: np.ndarray) -> float`**
   - Wrapper to evaluate log target with NumPy arrays
   - Convert to PyTorch, evaluate, convert back

3. **`_grad_log_target_numpy(x: np.ndarray) -> np.ndarray`**
   - Wrapper to compute gradient with NumPy arrays
   - Convert to PyTorch, compute gradient, convert back

4. **`_leapfrog(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**
   - Implement leapfrog integrator
   - Takes position q and momentum p
   - Returns new position and momentum after L steps

5. **`_hamiltonian(q: np.ndarray, p: np.ndarray) -> float`**
   - Compute total Hamiltonian: H(q,p) = U(q) + K(p)
   - U(q) = -log π(q) (potential energy)
   - K(p) = (1/2) p^T p (kinetic energy)

6. **`_propose(q: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, bool]`**
   - Generate HMC proposal:
     - Sample momentum: p ~ N(0, I)
     - Run leapfrog integrator
     - Compute acceptance probability using Hamiltonian values
     - Accept/reject with Metropolis criterion
   - Returns: (new_state, accepted)

7. **`_run_chain(...)`**
   - Run single MCMC chain
   - Handle burn-in and thinning
   - Store samples and statistics
   - Update progress bar

8. **`_setup_initial_states(...)`**
   - Initialize states for all chains
   - Handle different input formats
   - Default to standard normal if not provided

9. **`_create_inference_data(...)`**
   - Create ArviZ InferenceData object
   - Use existing `create_inference_data()` helper

10. **`get_acceptance_rates(idata: az.InferenceData) -> Dict[str, float]`**
    - Extract acceptance rates from InferenceData
    - Overall and per-chain rates

### Algorithm Flow

For each chain:
1. Initialize at starting state q₀
2. For each iteration:
   a. Sample momentum: p ~ N(0, I)
   b. Compute current Hamiltonian: H₀ = H(q, p)
   c. Run leapfrog integrator: (q, p) → (q*, p*)
   d. Negate momentum: p* → -p* (for reversibility)
   e. Compute proposed Hamiltonian: H* = H(q*, p*)
   f. Accept with probability: min(1, exp(H₀ - H*))
   g. Store sample if past burn-in and at thinning interval

## Code Standards

### Style Guidelines
- Python 3.12+ with strict typing
- Ruff formatting: 96-char lines, double quotes
- Google-style docstrings
- snake_case for functions/variables
- PascalCase for classes

### Documentation Requirements
- Comprehensive docstrings for class and all public methods
- Algorithm explanation in class docstring
- Parameter descriptions with types and defaults
- Return value descriptions
- Usage examples (1D and 2D cases)
- Notes section explaining HMC advantages and tuning
- References to key papers

### Example Usage (to include in docstring)

```python
# 1D Gaussian example
import torch

def log_normal(x):
    return -0.5 * x**2

sampler = HMCSampler(
    log_target=log_normal,
    step_size=0.2,
    n_leapfrog_steps=10
)
idata = sampler.sample(n_samples=1000, n_chains=4)

# 2D correlated Gaussian example
mean = torch.tensor([0.0, 0.0])
cov = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
cov_inv = torch.inverse(cov)

def log_target(x):
    diff = x - mean
    return -0.5 * torch.sum(diff * (cov_inv @ diff))

sampler = HMCSampler(
    log_target=log_target,
    step_size=0.15,
    n_leapfrog_steps=20,
    var_names=['x', 'y']
)
idata = sampler.sample(n_samples=2000, burn_in=500)
```

## Dependencies

All required dependencies already in `pyproject.toml`:
- `numpy` - Array operations
- `torch` - Automatic differentiation
- `arviz` - MCMC diagnostics
- `tqdm` - Progress bars

No new dependencies needed.

## Testing Strategy

Future testing should cover:
- Correctness: Sample from known distributions (1D/2D Gaussians)
- Statistical tests: Mean, variance, correlation structure
- Edge cases: Different dimensions, step sizes, leapfrog steps
- Numerical stability: Handle extreme values gracefully
- Acceptance rates: Should be in reasonable range (0.6-0.9 typically)

## Key Differences from MALA

**MALA (Langevin Dynamics):**
- First-order method (uses only gradients)
- Proposal includes noise: x' = x + (ε²/2)∇log π(x) + ε·Z
- Requires proposal correction in acceptance ratio
- Target acceptance: ~50-70%

**HMC (Hamiltonian Dynamics):**
- Second-order method (position + momentum)
- Deterministic dynamics (leapfrog integrator)
- No proposal correction (symmetric, volume-preserving)
- Target acceptance: ~60-90%
- Generally more efficient for complex distributions

## Educational Focus

The implementation should emphasize:
1. **Clarity over optimization** - Easy to understand algorithm
2. **Numerical stability** - Proper handling of log-probabilities
3. **Standard patterns** - Follows existing codebase conventions
4. **Comprehensive documentation** - Clear explanations of HMC concepts
5. **Practical examples** - Shows how to use for real problems

## References

Key papers to cite in docstrings:
1. Neal, R. M. (2011). "MCMC using Hamiltonian dynamics." Handbook of Markov Chain Monte Carlo.
2. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
