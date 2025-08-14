# mc-lab

Educational implementations of core Monte Carlo method algorithms. The goal is learning, clarity, and numerical correctness over raw speed or micro-optimizations. Expect straightforward NumPy/SciPy-based code with helpful tests and a couple of demo notebooks.

## What’s inside

- Basic importance sampling
- Rejection sampling and a few more advance rejection sampling methods
- Normal sampling via Box–Muller (classic and polar) — `src/mc_lab/box_mueller.py`
- Inverse transform sampling (analytical, numerical interpolation/root-finding, adaptive; plus alias method for discrete) — `src/mc_lab/inverse_transform.py`
- Multivariate Gaussian sampling with Cholesky/eigendecomposition fallback — `src/mc_lab/multivariate_gaussian.py`
- Tests in `tests/` and a demo notebook in `notebooks/`

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

With uv:

```bash
uv run pytest
# Format & lint (either use the Makefile below or direct commands)
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
# Or via the Makefile stored under src/mc_lab:
make -f src/mc_lab/Makefile format-fix
make -f src/mc_lab/Makefile lint-fix
```

## Quick usage example

```python
import numpy as np
from mc_lab.multivariate_gaussian import sample_multivariate_gaussian

mu = np.array([0.0, 1.0])
Sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
X = sample_multivariate_gaussian(mu, Sigma, n=1000, random_state=42)
print(X.shape)  # (1000, 2)
```

## Notebooks

Demo notebooks live in `notebooks/`. If you want the environment available as a Jupyter kernel:

```bash
python -m ipykernel install --user --name mc-lab
```

---

Note: This is an educational project; APIs and implementations may evolve for clarity. If you need production-grade performance, consider specialized libraries or contribute optimizations guarded by tests.

## Numba

The Box-Muller implementation can be used with numba for performance improvenments. See the installation notes for numba on how to set it up on your hardware or simply try

```bash
uv pip install numba
```

## Collaboration

Open to collaboration and contributions. If you’re interested:
- Open an issue to discuss ideas or report bugs.
- Submit small, focused PRs with tests when public behavior changes.
- For larger changes, start with a brief design proposal in an issue.

## Course context

This project was initiated and developed while following the course [“NMAK24010U Topics in Statistics”](https://kurser.ku.dk/course/nmak24010u/) at the University of Copenhagen (UCPH) fall 2025.
