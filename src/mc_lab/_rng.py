"""
RNG utilities: unified construction of numpy.random.Generator and a light
protocol for RNG-like objects used in this package.

This centralizes the logic for turning a random_state argument into a
np.random.Generator so all modules behave consistently.
"""

from __future__ import annotations

from typing import Optional, Protocol, Union, runtime_checkable

import numpy as np

__all__ = ["RNGLike", "RandomState", "as_generator"]


@runtime_checkable
class RNGLike(Protocol):
    """Minimal interface our sampling code relies on.

    This mirrors the subset of numpy.random.Generator methods we use.
    """

    def random(
        self, size: Optional[Union[int, tuple]] = None, /, **kwargs
    ) -> np.ndarray:  # noqa: D401
        ...

    def standard_normal(
        self, size: Optional[Union[int, tuple]] = None, /, **kwargs
    ) -> np.ndarray: ...

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Optional[Union[int, tuple]] = None,
        /,
        **kwargs,
    ) -> np.ndarray: ...


RandomState = Optional[Union[int, np.random.Generator]]


def as_generator(random_state: RandomState) -> np.random.Generator:
    """Return a NumPy Generator from an int seed, Generator, or None.

    None -> default bit generator.
    int  -> new PCG64 generator with the given seed.
    Generator -> returned as-is.
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is None:
        return np.random.default_rng()
    return np.random.default_rng(random_state)
