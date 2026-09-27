"""One cGA run as a unit of work: ``run_instance(spec) -> RunResult``.

Kept free of heavy imports (pandas, matplotlib) so that worker processes, which import only
this module and the simulator, start quickly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

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


def run_instance(spec: InstanceSpec, on_progress: Optional[Callable[[int], None]] = None) -> RunResult:
    """Run one cGA instance. Deterministic given spec (seeded), side-effect free, picklable.

    on_progress(iterations_done) is called periodically during the run (display only)."""
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
        on_progress=on_progress,
    )


# --------------------------------------------------------------------------- #
# Live progress from worker processes
# --------------------------------------------------------------------------- #
# The parent creates a shared array with SLOT_FIELDS numbers per worker. Each worker claims one
# slot when it starts and keeps (n, iterations done, budget) of its current run there; the parent
# reads the array to show progress while runs are still going. n = 0 marks an idle worker.

SLOT_FIELDS = 3
_slot: Optional[tuple] = None  # (shared array, slot index), set in worker processes only


def init_worker(shared, counter) -> None:
    """ProcessPoolExecutor initializer: claim a progress slot for this worker process."""
    global _slot
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    if (index + 1) * SLOT_FIELDS <= len(shared):
        _slot = (shared, index)


def timed_run(spec: InstanceSpec, on_progress: Optional[Callable[[int], None]] = None) -> tuple[RunResult, float]:
    """run_instance plus its wall time in seconds (what worker processes execute).

    In a worker with a progress slot, iterations done are published to the shared array.
    """
    if on_progress is None and _slot is not None:
        shared, index = _slot
        base = index * SLOT_FIELDS
        shared[base], shared[base + 1], shared[base + 2] = spec.n, 0, spec.max_iterations

        def on_progress(t: int) -> None:
            shared[base + 1] = t
    t0 = time.perf_counter()
    try:
        res = run_instance(spec, on_progress)
    finally:
        if _slot is not None:
            _slot[0][_slot[1] * SLOT_FIELDS] = 0  # idle
    return res, time.perf_counter() - t0
