"""Core cGA loop.

State after t updates: frequency vector p^(t) in [l, u]^n, with p^(0) = 1/2.
Iteration t (t = 1, 2, ...):
  1. Sample X, Y ~ p^(t-1) independently, bitwise independent.
  2. If X or Y is the all-ones optimum, stop: runtime T = t.
  3. Compare X, Y with the comparator. On a tie (None), p is unchanged.
  4. Otherwise, for each i: p_i += 1/K if W_i=1 and L_i=0,
     p_i -= 1/K if W_i=0 and L_i=1. Then clip p to [l, u].
Borders: l = 1/(L*n), u = 1 - l.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .comparators import Comparator


@dataclass(frozen=True)
class RunResult:
    converged: bool
    runtime: Optional[int]  # T (first iteration sampling the optimum), None if not converged
    iterations: int  # iterations actually executed (= T, or max_iterations)
    # Only filled if checkpoints were requested. checkpoint_times[j] = t means
    # checkpoint_p[j] = p^(t), the frequency vector after t updates (the
    # distribution sampled from in iteration t+1).
    checkpoint_times: Optional[np.ndarray] = None
    checkpoint_p: Optional[np.ndarray] = None


def borders(n: int, L: float) -> tuple[float, float]:
    lo = 1.0 / (L * n)
    return lo, 1.0 - lo


def run_cga(
    n: int,
    K: float,
    L: float,
    comparator: Comparator,
    max_iterations: int,
    rng: np.random.Generator,
    log_checkpoints: bool = False,
    on_progress: Optional[Callable[[int], None]] = None,
    progress_every: int = 1 << 16,
) -> RunResult:
    """Run the cGA once until the optimum is sampled or max_iterations is reached.

    With log_checkpoints, p^(t) is recorded at t = 0, 1, 2, 4, 8, ... (doubling),
    plus once more at the end of the run: at t = T-1 (the distribution the optimum
    was sampled from) if the run converged, or at t = max_iterations otherwise.

    on_progress(t), if given, is called every `progress_every` iterations with the number of
    iterations done so far (for live progress displays; it does not affect the run).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not K > 0:
        raise ValueError(f"K must be positive, got {K}")
    if not L > 0:
        raise ValueError(f"L must be positive, got {L}")
    if L < 1:
        warnings.warn(f"L = {L} < 1: the lower border 1/(L n) is wider than the standard 1/n")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

    lo, hi = borders(n, L)
    step = 1.0 / K
    p = np.full(n, 0.5)

    times: list[int] = []
    snaps: list[np.ndarray] = []
    next_ckpt = 1
    next_report = progress_every if on_progress is not None else max_iterations + 1
    if log_checkpoints:
        times.append(0)
        snaps.append(p.copy())

    def result(converged: bool, runtime: Optional[int], iterations: int) -> RunResult:
        if not log_checkpoints:
            return RunResult(converged, runtime, iterations)
        return RunResult(converged, runtime, iterations, np.asarray(times), np.vstack(snaps))

    for t in range(1, max_iterations + 1):
        XY = rng.random((2, n)) < p  # row 0 = X, row 1 = Y
        X, Y = XY[0], XY[1]

        if X.all() or Y.all():
            if log_checkpoints and times[-1] != t - 1:
                times.append(t - 1)
                snaps.append(p.copy())
            return result(True, t, t)

        outcome = comparator(X, Y)
        if outcome is not None:
            W, Lo = outcome
            # W_i - L_i is +1 (W_i=1, L_i=0), -1 (W_i=0, L_i=1) or 0 (bits agree).
            p += step * (W.astype(np.int8) - Lo.astype(np.int8))
            np.clip(p, lo, hi, out=p)

        if t == next_report:
            on_progress(t)
            next_report += progress_every

        if log_checkpoints and t == next_ckpt:
            times.append(t)
            snaps.append(p.copy())
            next_ckpt *= 2

    if log_checkpoints and times[-1] != max_iterations:
        times.append(max_iterations)
        snaps.append(p.copy())
    return result(False, None, max_iterations)
