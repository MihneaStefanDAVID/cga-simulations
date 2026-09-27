"""Fitness comparators for the cGA.

A *comparator* is a callable ``comparator(X, Y) -> (W, L) | None`` taking two
boolean bit vectors of length n. It returns the (winner, loser) pair, or
``None`` if the two individuals tie, in which case the cGA skips the update.

The simulator core only ever sees this callable, so which fitness function is
in use makes no difference to it. Comparators are built by factories
``factory(n, rng) -> comparator`` so that fitness functions needing
per-iteration randomness (e.g. Dynamic BinVal, which draws a fresh weight
permutation every iteration) can close over the run's random generator.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

Outcome = Optional[Tuple[np.ndarray, np.ndarray]]
Comparator = Callable[[np.ndarray, np.ndarray], Outcome]
ComparatorFactory = Callable[[int, np.random.Generator], Comparator]


def binval_comparator(X: np.ndarray, Y: np.ndarray) -> Outcome:
    """Compare X and Y under static BinVal(x) = sum_i 2^(n-i) x_i.

    Bit 1 (array index 0) is the most significant bit. Comparing two strings
    under BinVal is the same as comparing them lexicographically: the first
    position h where they differ decides, because 2^(n-h) > sum_{j>h} 2^(n-j).
    The individual with a 1 at position h wins. If X == Y there is no
    differing bit, and the result is a tie (None).
    """
    diff = X != Y
    h = int(diff.argmax())  # first True, or 0 if there is none
    if not diff[h]:
        return None
    return (X, Y) if X[h] else (Y, X)


def _binval_factory(n: int, rng: np.random.Generator) -> Comparator:
    return binval_comparator


def _not_implemented_factory(name: str, note: str) -> ComparatorFactory:
    def factory(n: int, rng: np.random.Generator) -> Comparator:
        raise NotImplementedError(f"fitness '{name}' is not implemented yet. {note}")

    return factory


# Registry: YAML field `fitness: <key>` selects the factory.
# To add a fitness function, write a factory with signature (n, rng) -> comparator
# and register it here.
_FACTORIES: dict[str, ComparatorFactory] = {
    "binval": _binval_factory,
    "onemax": _not_implemented_factory(
        "onemax",
        "Decide the tie policy first: X != Y with |X|_1 == |Y|_1 is a tie "
        "(return None), or break it at random with rng?",
    ),
    "dynamic_binval": _not_implemented_factory(
        "dynamic_binval",
        "Draw a fresh uniformly random permutation of the weights with rng in every "
        "call, then compare lexicographically in permuted order.",
    ),
}


def available_fitness_functions() -> list[str]:
    return sorted(_FACTORIES)


def make_comparator(name: str, n: int, rng: np.random.Generator) -> Comparator:
    """Build the comparator registered under ``name`` for dimension n."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"unknown fitness '{name}'; available: {available_fitness_functions()}"
        ) from None
    return factory(n, rng)
