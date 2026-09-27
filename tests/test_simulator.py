"""Correctness tests. Run with `python tests/test_simulator.py` or `pytest tests/`."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cga.comparators import binval_comparator  # noqa: E402
from cga.experiments import InstanceSpec, run_instance  # noqa: E402
from cga.expressions import eval_expr  # noqa: E402
from cga.simulator import borders, run_cga  # noqa: E402


def binval_int(x: np.ndarray) -> int:
    """Reference BinVal as an exact Python integer (bit 1 = most significant)."""
    n = len(x)
    return sum(int(b) << (n - 1 - i) for i, b in enumerate(x))


def test_binval_matches_integer_value_exhaustively():
    for n in range(1, 7):
        strings = [np.array(s, dtype=bool) for s in itertools.product([0, 1], repeat=n)]
        for X, Y in itertools.product(strings, repeat=2):
            out = binval_comparator(X, Y)
            fx, fy = binval_int(X), binval_int(Y)
            if fx == fy:
                assert out is None
            else:
                W, L = out
                assert binval_int(W) == max(fx, fy) and binval_int(L) == min(fx, fy)


def test_binval_large_n_random():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(61, 300))  # beyond int64 range
        X = rng.random(n) < 0.5
        Y = X.copy()
        k = int(rng.integers(0, n))
        Y[k:] = rng.random(n - k) < 0.5
        out = binval_comparator(X, Y)
        fx, fy = binval_int(X), binval_int(Y)
        assert (out is None) == (fx == fy)
        if out is not None:
            assert binval_int(out[0]) == max(fx, fy)


class ScriptedRng:
    """Feeds prescribed uniforms to run_cga, to check a single update exactly."""

    def __init__(self, rows):
        self.rows = list(rows)

    def random(self, shape):
        return np.asarray(self.rows.pop(0), dtype=float).reshape(shape)


def test_single_update_rule_and_clipping():
    n, K, L = 4, 4.0, 1.0
    lo, hi = borders(n, L)
    # p = 0.5 everywhere: uniform < 0.5 gives bit 1.
    # X = 1 0 1 0, Y = 0 1 1 0 -> first difference at bit 1, X wins.
    u = [[0.1, 0.9, 0.1, 0.9], [0.9, 0.1, 0.1, 0.9]]
    # Second iteration samples the optimum (all uniforms 0).
    rng = ScriptedRng([u, np.zeros((2, n))])
    res = run_cga(n, K, L, binval_comparator, 10, rng, log_checkpoints=True)
    assert res.converged and res.runtime == 2
    p1 = res.checkpoint_p[list(res.checkpoint_times).index(1)]
    np.testing.assert_allclose(p1, [0.75, 0.25, 0.5, 0.5])
    assert lo <= p1.min() and p1.max() <= hi

    # Clipping: step 1/K = 1 pushes the winner's bits beyond the borders.
    rng = ScriptedRng([u, np.zeros((2, n))])
    res = run_cga(n, 1.0, L, binval_comparator, 10, rng, log_checkpoints=True)
    p1 = res.checkpoint_p[list(res.checkpoint_times).index(1)]
    np.testing.assert_allclose(p1, [hi, lo, 0.5, 0.5])


def test_tie_is_noop():
    n = 3
    u = [[0.9, 0.1, 0.9], [0.9, 0.1, 0.9]]  # X == Y = 0 1 0
    rng = ScriptedRng([u, np.zeros((2, n))])
    res = run_cga(n, 2.0, 1.0, binval_comparator, 10, rng, log_checkpoints=True)
    np.testing.assert_allclose(res.checkpoint_p[1], 0.5)


def test_budget_and_determinism():
    spec = InstanceSpec(n=100, K=2.0, L=1.0, fitness="binval", max_iterations=50, seed=1)
    res = run_instance(spec)
    assert not res.converged and res.runtime is None and res.iterations == 50
    spec = InstanceSpec(n=10, K=10.0, L=1.0, fitness="binval", max_iterations=100_000, seed=3,
                        log_checkpoints=True)
    a, b = run_instance(spec), run_instance(spec)
    assert a.converged and a.runtime == b.runtime
    np.testing.assert_array_equal(a.checkpoint_p, b.checkpoint_p)
    t = list(a.checkpoint_times)
    assert t[:4] == [0, 1, 2, 4] and t[-1] == a.runtime - 1


def test_frequencies_stay_in_borders():
    n, L = 20, 2.0
    lo, hi = borders(n, L)
    res = run_instance(InstanceSpec(n, 3.0, L, "binval", 5000, 7, log_checkpoints=True))
    P = res.checkpoint_p[1:]  # skip the initial 0.5
    assert P.min() >= lo - 1e-12 and P.max() <= hi + 1e-12


def test_eval_expr():
    assert abs(eval_expr("5 * log(n)", 100) - 5 * np.log(100)) < 1e-12
    assert eval_expr(50, 100) == 50.0
    assert eval_expr("0.3*n", 10) == 3.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
