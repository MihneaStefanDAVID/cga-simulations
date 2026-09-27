"""One cGA run as a unit of work: ``run_instance(spec) -> RunResult``.

Kept free of heavy imports (pandas, matplotlib) so that worker processes, which import only
this module and the simulator, start quickly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .comparators import make_comparator
from .simulator import RunResult, run_cga


@dataclass(frozen=True)
class InstanceSpec:
    n: int
    K: float
    L: float
    fitness: str
    max_iterations: int
    seed: int
    log_checkpoints: bool = False


def run_instance(spec: InstanceSpec) -> RunResult:
    """Run one cGA instance. Deterministic given spec (seeded), side-effect free, picklable."""
    rng = np.random.default_rng(spec.seed)
    comparator = make_comparator(spec.fitness, spec.n, rng)
    return run_cga(
        n=spec.n,
        K=spec.K,
        L=spec.L,
        comparator=comparator,
        max_iterations=spec.max_iterations,
        rng=rng,
        log_checkpoints=spec.log_checkpoints,
    )


def timed_run(spec: InstanceSpec) -> tuple[RunResult, float]:
    """run_instance plus its wall time in seconds (what worker processes execute)."""
    t0 = time.perf_counter()
    res = run_instance(spec)
    return res, time.perf_counter() - t0
