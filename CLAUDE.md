# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MC-LAB is an educational repository for learning Monte Carlo statistical methods through clear, numerical Python implementations. The codebase prioritizes learning, clarity, and correctness over raw performance.

### Core Architecture

**Central RNG System**: All modules use a unified random number generator interface through `_rng.py`:

- `RandomState` type alias accepts int seeds, Generator objects, or None
- `as_generator()` function normalizes inputs to `np.random.Generator`
- `RNGLike` protocol defines the minimal interface used across modules

**Module Organization**: Each sampling method is implemented as a standalone module with:

- Main sampling function(s)
- Helper utilities specific to that method
- Comprehensive docstrings and type hints
- Corresponding test file and demonstration notebook

**Key Modules**:

- `box_muller.py` - Normal sampling (classic/polar methods, optional Numba support)
- `inverse_transform.py` - Analytical, numerical, and discrete inverse transforms
- `fast_inverse_transform.py` - Optimized inverse sampling implementations  
- `rejection_sampling.py` + `advanced_rejection_sampling.py` - Basic and TDR methods
- `multivariate_gaussian.py` - Cholesky/eigendecomposition with fallback
- `importance_sampling.py` - Weighted sampling with diagnostics
- `gibbs_sampler_2d.py` - 2D Gibbs sampling for correlated variables
- `PSIS.py` - Pareto Smoothed Importance Sampling
- `transformation_methods.py` - Various transformation-based samplers

## Essential Commands

### Development Workflow

```bash
uv sync                    # Setup environment
source .venv/bin/activate  # Activate environment
```

### Testing and Quality

```bash
make test          # Run all non-performance tests
make perftest      # Run performance tests with output
make format-fix    # Format code and sort imports
make lint-fix      # Lint and auto-fix issues
```

### Individual Test Execution

```bash
uv run pytest tests/test_box_muller.py::test_specific_function
uv run pytest -k "pattern"
```

## Development Guidelines

**Dependencies**: Always ask before adding new packages. Current stack: NumPy, SciPy, matplotlib, pytest, ruff.

**RNG Usage**: All functions accepting randomness should use the `RandomState` type and `as_generator()` utility from `_rng.py`.

**Testing Strategy**:

- Statistical correctness over micro-benchmarks
- Performance tests marked with `@pytest.mark.performance`
- Separate test files mirror source structure

**Notebooks**: Each major algorithm has a corresponding demo notebook showing usage and visualizations. No trailing newlines in notebook files. Follow these rules when created and updating notebooks:

- Rule 1: Tell a story for an audience
- Rule 2: Document the process, not just the results
- Rule 3: Use cell divisions to make steps clear
- Rule 4: Modularize code
- Rule 5: Design your notebooks to be read, run, and explored

Use the following colors for visualization:
#332288
#117733
#44AA99
#88CCEE
#DDCC77
#CC6677
#AA4499
#AA4499


## Code Standards

- Python 3.12+ with strict typing
- Ruff formatting: 96-char lines, double quotes, sorted imports
- Google-style docstrings for public functions/classes
- snake_case functions/variables, PascalCase classes
- error Handling - use typed exceptions
- documentation: Google-style docstrings for public functions/classes
  