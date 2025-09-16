# mc-lab

Educational implementations of core Monte Carlo method algorithms. The goal is learning, clarity, and correctness over raw speed or micro-optimizations. Expect straightforward NumPy/SciPy-based code with helpful tests and comprehensive demo notebooks.

## Collaboration

Open to collaboration and contributions. If you're interested:

- Open an issue to discuss ideas or report bugs.
- Submit small, focused PRs with tests when public behavior changes.
- For larger changes, start with a brief design proposal in an issue.

## Publishing new versions to PyPI

This package is published to PyPI as `mc-lab-edu`. To publish a new version:

### 1. Update the version

Edit `pyproject.toml` and increment the version number:

```toml
[project]
name = "mc-lab-edu"
version = "0.1.1"  # or 0.2.0 for larger changes
```

### 2. Build and upload

```bash
# Clean previous builds
rm -rf dist/

# Build the package
uv run python -m build

# Check the package for issues
uv run twine check dist/*

# Upload to PyPI (requires API token in ~/.pypirc)
uv run twine upload dist/*
```

### 3. Setup PyPI authentication (one-time setup)

If you haven't set up authentication yet:

1. Get an API token from <https://pypi.org/manage/account/token/>
2. Create `~/.pypirc`:

```ini
[distutils]
index-servers = pypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-your-actual-token-here
```

**Note**: Never commit API tokens to version control. The `.pypirc` file should remain in your home directory only.

## What's inside

### Core Sampling Methods

- **Box-Muller Sampling** — Normal distribution sampling (classic and polar methods, optional Numba support) — `box_muller.py`
- **Inverse Transform Sampling** — Analytical, numerical, and discrete inverse transforms — `inverse_transform.py` and `fast_inverse_transform.py`
- **Rejection Sampling** — Basic and advanced methods including Transformed Density Rejection (TDR) — `rejection_sampling.py` and `advanced_rejection_sampling.py`
- **Importance Sampling** — Weighted sampling with diagnostics and Pareto Smoothed Importance Sampling (PSIS) — `importance_sampling.py` and `PSIS.py`
- **Multivariate Gaussian** — Cholesky/eigendecomposition with fallback — `multivariate_gaussian.py`
- **MCMC Methods** — Gibbs sampling, Metropolis-Hastings, and Independent Metropolis-Hastings — `gibbs_sampler_2d.py`, `metropolis_hastings.py`, `independent_metropolis_hastings.py`
- **Transformation Methods** — Various transformation-based samplers — `transformation_methods.py`

### Utilities

- **Unified RNG System** — Central random number generator interface — `_rng.py`
- **Statistical Diagnostics** — Distribution and moment comparison tools — `distribution_comparison.py`, `moment_comparison.py`
- **Visualization** — Plotting utilities — `plotting.py`

### Documentation

- Comprehensive tests in `tests/`
- Demo notebooks in `notebooks/` for each major algorithm
- Type hints and Google-style docstrings throughout

## Installation

Install from PyPI:

```bash
pip install mc-lab-edu
```

### Google Colab Compatibility

Version 0.2.2+ has been specifically tested to work with Google Colab's package environment. The numpy version is constrained to `>=1.24.0,<2.0.0` to avoid compatibility issues with pre-installed packages and scipy. This resolves the `_center` import errors that occurred with numpy 2.x versions.

Or for local development:

## Clone the repository

```bash
git clone https://github.com/carsten-j/mc-lab.git
cd mc-lab
```

## Setup for local development

Recommended (uses uv to manage a local .venv and sync dependencies):

```bash
# If you don’t have uv yet, see https://docs.astral.sh/uv/ for install options
uv sync
source .venv/bin/activate
```

Alternative (standard venv + pip):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
# Dev tools (optional but recommended)
pip install pytest ruff ipykernel pre-commit
```

Verify the install:

```bash
python -c "import mc_lab; print(mc_lab.hello())"
```

## Run tests and linting

### Essential Commands

```bash
# Setup and development
uv sync                    # Setup environment
source .venv/bin/activate  # Activate environment

# Testing and quality
make test          # Run all non-performance tests
make perftest      # Run performance tests with output
make format-fix    # Format code and sort imports
make lint-fix      # Lint and auto-fix issues
```

### Alternative commands (without Makefile)

With uv:

```bash
uv run pytest
uv run ruff format .
uv run ruff check --select I --fix
uv run ruff check --fix
```

With a plain venv:

```bash
pytest
ruff format .
ruff check --select I --fix
ruff check --fix
```

## Quick usage examples

### Basic sampling

```python
import numpy as np
from mc_lab.multivariate_gaussian import sample_multivariate_gaussian
from mc_lab.box_muller import box_muller_polar
from mc_lab.importance_sampling import importance_sample

# Multivariate Gaussian
mu = np.array([0.0, 1.0])
Sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
X = sample_multivariate_gaussian(mu, Sigma, n=1000, random_state=42)
print(X.shape)  # (1000, 2)

# Box-Muller normal sampling
samples = box_muller_polar(n=1000, random_state=42)
print(f"Mean: {np.mean(samples):.3f}, Std: {np.std(samples):.3f}")
```

### MCMC sampling

```python
from mc_lab.metropolis_hastings import metropolis_hastings
from mc_lab.gibbs_sampler_2d import gibbs_sample_2d

# Metropolis-Hastings for standard normal
def log_density(x):
    return -0.5 * np.sum(x**2)

samples = metropolis_hastings(log_density, initial_state=0.0, n_samples=1000)
```

## Demo Notebooks

Comprehensive demo notebooks are available in `notebooks/`:

- `inverse_transform_demo.ipynb` — Inverse transform techniques
- `importance_sampling_demo.ipynb` — Importance sampling with diagnostics
- `metropolis_hastings_demo.ipynb` — MCMC methods
- `gibbs_sampler_demo.ipynb` — Gibbs sampling
- `mv_gaussian_demo.ipynb` — Multivariate Gaussian sampling
- `tdr_demo.ipynb` — Transformed Density Rejection
- `transformation_methods_demo.ipynb` — Various transformation methods
- And many more specialized demonstrations

To use the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name mc-lab
```

---

## Performance Notes

### Numba Support

The Box-Muller implementation supports optional Numba acceleration for performance improvements:

```bash
uv pip install numba
```

See the Numba installation documentation for hardware-specific setup instructions.

### Design Philosophy

This codebase prioritizes:

1. **Learning and clarity** over raw performance
2. **Correctness** over micro-optimizations
3. **Educational value** with comprehensive documentation
4. **Unified design** with consistent RNG interfaces across all modules

## Architecture

### Central RNG System

All modules use a unified random number generator interface through `_rng.py`:

- `RandomState` type alias accepts int seeds, Generator objects, or None
- `as_generator()` function normalizes inputs to `np.random.Generator`
- `RNGLike` protocol defines the minimal interface used across modules

### Module Design

Each sampling method follows a consistent pattern:

- Main sampling function(s) with comprehensive type hints
- Helper utilities specific to that method
- Google-style docstrings with mathematical background
- Corresponding test file and demonstration notebook

## Collaboration

Open to collaboration and contributions. If you're interested:

- Open an issue to discuss ideas or report bugs
- Submit small, focused PRs with tests when public behavior changes
- For larger changes, start with a brief design proposal in an issue

Note: This is an educational project; APIs and implementations may evolve for clarity. If you need production-grade performance, consider specialized libraries or contribute optimizations guarded by tests.

## Course context

This project was initiated and developed while following the course [“NMAK24010U Topics in Statistics”](https://kurser.ku.dk/course/nmak24010u/) at the University of Copenhagen (UCPH) fall 2025.
