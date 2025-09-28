# Barker MCMC Implementation

This document describes the implementation of Barker's MCMC algorithm in the `mc-lab` package.

## Algorithm Overview

Barker's MCMC algorithm is an alternative to the standard Metropolis-Hastings algorithm with a different acceptance rule. According to the mathematical specification:

**Barker's MCMC Algorithm:**
1. Given current state Xₙ, draw X* ~ r(·|Xₙ) from the proposal distribution
2. Set Xₙ₊₁ = X* with probability π(X*)/(π(X*) + π(Xₙ)), otherwise Xₙ₊₁ = Xₙ

Where:
- π(x) is the target distribution (unnormalized)
- r(·|·) is the symmetric proposal transition density (Gaussian in our implementation)

## Key Differences from Metropolis-Hastings

| Aspect | Metropolis-Hastings | Barker MCMC |
|--------|---------------------|-------------|
| Acceptance probability | min(1, π(X*)/π(Xₙ)) | π(X*)/(π(X*) + π(Xₙ)) |
| Optimal acceptance rate | ~0.35 | ~0.5 |
| Mixing properties | Standard | Often better for multimodal distributions |

## Implementation Details

### File Structure
```
src/mc_lab/barker_mcmc.py          # Main implementation
tests/test_barker_mcmc.py          # Comprehensive tests  
scripts/barker_mcmc_demo.py        # Demo script
```

### Core Features

1. **Gaussian Proposal Distribution**: Uses N(0, σ²I) as assumed in the specification
2. **Adaptive Proposal Scaling**: Automatically tunes σ during burn-in to achieve target acceptance rate
3. **ArviZ Integration**: Returns standardized `InferenceData` objects for diagnostics
4. **Multi-chain Support**: Runs multiple independent chains for convergence assessment
5. **Numerical Stability**: Uses log-sum-exp trick for computing acceptance probabilities

### Usage Example

```python
from mc_lab import BarkerMCMCSampler
import numpy as np

# Define target log-probability function
def log_target(x):
    x = np.asarray(x).flatten()
    return float(-0.5 * x[0]**2)  # Standard normal

# Create sampler
sampler = BarkerMCMCSampler(
    log_target=log_target,
    proposal_scale=0.8,
    target_acceptance_rate=0.5,  # Higher than MH optimal
)

# Generate samples
idata = sampler.sample(
    n_samples=1000,
    n_chains=2,
    burn_in=500,
    random_seed=42,
)

# Access results
samples = idata.posterior.x.values
acceptance_rate = sampler.get_acceptance_rates(idata)["overall"]
```

### Implementation Formula

The acceptance probability is computed as:
```python
# In log space using log-sum-exp for numerical stability
log_sum = np.logaddexp(proposal_log_prob, current_log_prob)
barker_acceptance_prob = np.exp(proposal_log_prob - log_sum)
```

This is mathematically equivalent to:
```python
# Direct computation (less numerically stable)
barker_acceptance_prob = (
    np.exp(proposal_log_prob) / 
    (np.exp(proposal_log_prob) + np.exp(current_log_prob))
)
```

### Testing

The implementation includes comprehensive tests covering:
- 1D and 2D distributions
- Adaptive scaling behavior
- Acceptance rate tracking
- Barker-specific diagnostics
- Error handling for invalid inputs
- Statistical accuracy validation

Run tests with:
```bash
pytest tests/test_barker_mcmc.py -v
```

### Performance Characteristics

- **Acceptance Rate**: Typically converges to ~0.5 (vs ~0.35 for MH)
- **Mixing**: Often superior to MH for complex posterior geometries
- **Computational Cost**: Similar to standard MH per iteration
- **Memory Usage**: Tracks additional acceptance probability statistics

### Integration with Existing Code

The Barker sampler follows the same interface patterns as `MetropolisHastingsSampler`:
- Same parameter names and types
- Compatible with ArviZ workflow
- Identical return format (`InferenceData`)
- Consistent error handling

Both samplers are available via:
```python
from mc_lab import BarkerMCMCSampler, MetropolisHastingsSampler
```

### Mathematical Verification

The implementation correctly follows the Barker formula. For example:
- Current state: x = 1.0, log π(x) = -0.5
- Proposal: x* = 0.5, log π(x*) = -0.125
- Barker acceptance: exp(-0.125) / (exp(-0.125) + exp(-0.5)) = 0.593
- MH acceptance: min(1, exp(-0.125 - (-0.5))) = 1.0

This demonstrates Barker's more conservative acceptance behavior.
