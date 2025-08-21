# Metropolis-Hastings Algorithm Implementation Plan

## Overview

This document outlines the implementation plan for the Metropolis-Hastings MCMC algorithm in MC-LAB, following the project's educational approach with clear, numerical Python implementations that prioritize learning, clarity, and correctness over raw performance.

## Algorithm Background

The Metropolis-Hastings algorithm is a Markov Chain Monte Carlo (MCMC) method for sampling from probability distributions where direct sampling is difficult. Given a target distribution π(x), the algorithm generates a sequence of samples by:

1. **Proposal Step**: From current state x_t, propose a new state x' using a proposal distribution q(x'|x_t)
2. **Acceptance Step**: Accept x' with probability α = min(1, [π(x') × q(x_t|x')] / [π(x_t) × q(x'|x_t)])
3. **Update Step**: Set x_{t+1} = x' if accepted, otherwise x_{t+1} = x_t

For the Random Walk Metropolis (our focus), the proposal is symmetric: q(x'|x_t) = q(x_t|x'), simplifying the acceptance probability to α = min(1, π(x')/π(x_t)).

## Implementation Architecture

### Core Class: `MetropolisHastingsSampler`

```python
class MetropolisHastingsSampler:
    def __init__(
        self,
        log_target: Callable,
        proposal_scale: Union[float, np.ndarray] = 1.0,
        var_names: Optional[List[str]] = None,
        adaptive_scaling: bool = True,
        target_acceptance_rate: float = 0.35,
    )
```

### Key Features

1. **Random Walk Proposal Only**: Simplifies the implementation and focuses on the most commonly used variant
2. **Log-Probability Interface**: Uses log π(x) for numerical stability with extreme probabilities
3. **ArviZ Integration**: Full compatibility with ArviZ InferenceData for comprehensive diagnostics
4. **Adaptive Scaling**: Automatic tuning of proposal step size for optimal acceptance rates
5. **Multi-dimensional Support**: Handles scalar and vector parameters with independent or correlated proposals
6. **Multiple Chains**: Supports multiple independent chains for convergence diagnostics

### Proposal Distribution

**Random Walk Metropolis**: 
- Proposal: x' = x_t + ε, where ε ~ N(0, σ²I)
- Symmetric: q(x'|x_t) = q(x_t|x')
- Step size σ controls exploration vs exploitation trade-off
- Optimal acceptance rate: ~23.4% (1D) to ~44% (high-D)

### Adaptive Scaling Algorithm

During burn-in period:
1. Track acceptance rate in batches (e.g., every 50-100 iterations)
2. If acceptance rate > target + tolerance: increase σ by factor (e.g., 1.1)
3. If acceptance rate < target - tolerance: decrease σ by factor (e.g., 0.9)
4. Stop adaptation after burn-in to ensure proper convergence

## ArviZ Integration Pattern

Following the `GibbsSampler2D` approach:

### Output Structure
```python
def sample(...) -> az.InferenceData:
    # Returns InferenceData with:
    # - posterior: (chain, draw, *param_dims)
    # - sample_stats: log_likelihood, acceptance_rate, proposal_scale
    # - metadata: sampling_method, burn_in, thin, etc.
```

### Sample Statistics
- `log_likelihood`: log π(x) for each sample
- `accepted`: Boolean indicator for each proposal
- `acceptance_rate`: Running acceptance rate per chain
- `proposal_scale`: Current step size (if adaptive)

### Coordinates and Dimensions
- `chain`: [0, 1, ..., n_chains-1]
- `draw`: [0, 1, ..., n_samples-1]
- Parameter dimensions: automatic detection from target function output

## API Design

### Main Sampling Method
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

### Parameters
- `n_samples`: Samples per chain (after burn-in and thinning)
- `n_chains`: Number of independent chains for convergence diagnostics
- `burn_in`: Samples to discard for chain equilibration
- `thin`: Keep every nth sample to reduce autocorrelation
- `initial_states`: Starting values, auto-generated if None
- `random_seed`: For reproducibility
- `progressbar`: Show sampling progress with tqdm

### Helper Methods
```python
def tune_proposal_scale(self, target_samples: int = 1000) -> float
def get_acceptance_rate(self) -> Dict[str, float]
def reset_adaptation(self) -> None
```

## File Structure

### Source Implementation
`src/mc_lab/metropolis_hastings.py`
- Main `MetropolisHastingsSampler` class
- Utility functions for proposal tuning
- Integration with `_rng.py` system
- Comprehensive docstrings with mathematical details

### Test Suite  
`tests/test_metropolis_hastings.py`
- Statistical correctness tests against known distributions
- ArviZ integration validation
- Convergence diagnostics verification
- Edge cases and error handling
- Performance benchmarks (marked with `@pytest.mark.performance`)

### Educational Demo
`notebooks/metropolis_hastings_demo.ipynb`
- Algorithm theory and intuition
- Step-by-step implementation walkthrough
- Comparison of different step sizes and their effects
- ArviZ diagnostic tutorial (trace plots, ESS, R-hat)
- Challenging sampling problems (multimodal, heavy-tailed)
- Comparison with Gibbs sampling trade-offs

## Testing Strategy

### Statistical Validation
1. **Known Distributions**: Sample from standard distributions (normal, exponential, beta) and validate moments
2. **Multivariate Normal**: Compare sample covariance with theoretical values
3. **Convergence Tests**: Multiple chains with R-hat < 1.01 for simple problems
4. **Acceptance Rates**: Verify adaptive tuning achieves target rates

### ArviZ Compatibility
1. **InferenceData Structure**: Validate all required groups and coordinates
2. **Diagnostic Functions**: Ensure compatibility with az.plot_trace, az.ess, az.rhat, etc.
3. **Sample Statistics**: Verify log_likelihood and acceptance tracking

### Edge Cases
1. **Boundary Conditions**: Constrained parameters, boundary reflections
2. **Initialization**: Extreme starting values, multiple chain initialization
3. **Error Handling**: Invalid target functions, numerical issues
4. **Performance**: Large sample sizes, high-dimensional problems

## Educational Content

### Notebook Structure
1. **Introduction**: MCMC motivation, when to use Metropolis-Hastings
2. **Algorithm Details**: Step-by-step breakdown with visualizations
3. **Basic Examples**: 1D normal distribution with different step sizes
4. **Proposal Tuning**: Interactive exploration of acceptance rates
5. **Multivariate Example**: 2D correlated normal with trace plots
6. **Challenging Cases**: Banana-shaped distribution, mixture models
7. **Diagnostics Tutorial**: Complete ArviZ workflow
8. **Best Practices**: Chain initialization, convergence assessment

### Learning Objectives
- Understand the accept/reject mechanism
- Appreciate the balance between exploration and exploitation
- Learn to diagnose convergence and mixing problems
- Compare MCMC methods (MH vs Gibbs) for different problems
- Master ArviZ for MCMC diagnostics

## Implementation Notes

### RNG Integration
- Use `RandomState` type from `_rng.py`
- Call `as_generator()` for consistent random number generation
- Support both seeded and unseeded sampling

### Code Quality
- Follow MC-LAB patterns: Google-style docstrings, comprehensive type hints
- Ruff formatting: 96-char lines, double quotes, sorted imports
- Statistical correctness over micro-optimizations
- Clear variable names and extensive comments for educational value

### Performance Considerations
- Vectorized operations where possible without sacrificing clarity
- Efficient memory allocation for large sample sizes
- Optional Numba acceleration (marked clearly as advanced feature)
- Progress bars for long-running samplers

## Success Criteria

1. **Functional**: Correctly samples from target distributions with proper statistics
2. **Educational**: Clear, understandable implementation with excellent documentation
3. **Integrated**: Full ArviZ compatibility for professional MCMC workflows
4. **Robust**: Handles edge cases gracefully with informative error messages
5. **Tested**: Comprehensive test coverage with statistical validation
6. **Consistent**: Follows all MC-LAB conventions and patterns

This implementation will provide MC-LAB users with a solid foundation in MCMC methods while maintaining the project's commitment to educational clarity and statistical rigor.