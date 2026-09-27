"""Compact Genetic Algorithm (cGA) simulator for runtime-analysis experiments."""

from .comparators import binval_comparator, make_comparator, available_fitness_functions
from .simulator import RunResult, borders, run_cga

__all__ = [
    "RunResult",
    "available_fitness_functions",
    "binval_comparator",
    "borders",
    "make_comparator",
    "run_cga",
]
