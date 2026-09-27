"""Correctness tests. Run with `python tests/test_simulator.py` or `pytest tests/`."""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cga.analysis import fit_power_law, scaling_fits  # noqa: E402
from cga.comparators import binval_comparator  # noqa: E402
from cga.experiments import (KEY_COLUMNS, InstanceSpec, geometric_ns, load_rep, parse_experiment,  # noqa: E402
                             resolve_workers, run_experiment, run_instance)
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


def test_geometric_ns():
    assert geometric_ns(100, 3200, factor=2) == [100, 200, 400, 800, 1600, 3200]
    assert geometric_ns(100, 1000, factor=2) == [100, 200, 400, 800]
    assert geometric_ns(10, 1000, count=3) == [10, 100, 1000]


def test_fit_power_law_recovers_exponent():
    n = np.array([100, 200, 400, 800, 1600], dtype=float)
    b, a, r2 = fit_power_law(n, 3.0 * n ** 1.5)
    assert abs(b - 1.5) < 1e-9 and abs(np.exp(a) - 3.0) < 1e-6 and abs(r2 - 1) < 1e-12
    assert fit_power_law(n[:1], n[:1]) is None


def test_scaling_fit_uses_only_uncensored_points():
    import pandas as pd
    n = [100, 200, 400, 800]
    summary = pd.DataFrame({
        "n": n, "K_expr": ["k"] * 4,
        "median_censored": [1e3, 4e3, 1.6e4, np.nan],  # T ~ n^2 where > half finished
        "runtime_median": [1e3, 4e3, 1.6e4, 5e3],      # biased point at n = 800 must be ignored
    })
    fit = scaling_fits(summary, ["k"]).iloc[0]
    assert abs(fit.exponent_b - 2.0) < 1e-9 and fit.n_points_used == 3 and fit.n_max_used == 400


def _scaling_entry(**over):
    e = {"name": "s", "type": "scaling", "n": {"from": 10, "to": 40, "factor": 2}, "K_values": ["2*log(n)"],
         "L": 2.0, "repetitions": 3, "max_iterations": 50_000, "seed": 1, "normalize_by": "n*log(n)"}
    e.update(over)
    return e


def test_scaling_config_validation():
    cfg = parse_experiment(_scaling_entry())
    assert cfg.ns == (10, 20, 40) and cfg.K_exprs == ("2*log(n)",) and cfg.normalize_by == "n*log(n)"
    for bad in [{"n": [50]}, {"n": {"from": 10, "to": 40}}, {"n": {"from": 10, "to": 40, "factor": 1}},
                {"normalize_by": "n - 100"}, {"K_values": []}]:
        try:
            parse_experiment(_scaling_entry(**bad))
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid scaling config {bad}")


def test_scaling_pipeline(tmp_path=None):
    import tempfile
    root = Path(tmp_path or tempfile.mkdtemp())
    out = run_experiment(_scaling_entry(), root, log=lambda _m: None)
    for f in ["raw_results.csv", "summary.csv", "scaling_fits.csv", "runtime_vs_n.png",
              "runtime_normalized.png", "experiment.yaml"]:
        assert (out / f).exists(), f


def _parallel_workers() -> int:
    return min(4, os.cpu_count() or 1)


def _raw_sorted(out: Path):
    import pandas as pd
    raw = pd.read_csv(out / "raw_results.csv", dtype={"K_expr": str})
    return raw.drop(columns="wall_seconds").sort_values(KEY_COLUMNS).reset_index(drop=True)


def test_parallel_grid_matches_sequential():
    import tempfile
    import pandas as pd
    w = _parallel_workers()
    if w < 2:
        print("  (skipped: fewer than 2 CPUs)")
        return
    entry = {"name": "par", "type": "sweep", "n": [12, 24], "K_values": ["2*log(n)", "n"], "L": 1.0,
             "repetitions": 4, "max_iterations": 20_000, "seed": 3}
    seq = run_experiment(entry, Path(tempfile.mkdtemp()), log=lambda _m: None, workers=1)
    par = run_experiment(entry, Path(tempfile.mkdtemp()), log=lambda _m: None, workers=w)
    a, b = _raw_sorted(seq), _raw_sorted(par)
    assert len(a) == 16
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(pd.read_csv(seq / "summary.csv"), pd.read_csv(par / "summary.csv"))


def test_parallel_trajectory_matches_sequential():
    import tempfile
    w = _parallel_workers()
    if w < 2:
        print("  (skipped: fewer than 2 CPUs)")
        return
    entry = {"name": "partraj", "type": "trajectory", "n": 20, "K": "3*log(n)", "L": 1.0, "repetitions": 4,
             "max_iterations": 20_000, "seed": 5}
    seq = run_experiment(entry, Path(tempfile.mkdtemp()), log=lambda _m: None, workers=1)
    par = run_experiment(entry, Path(tempfile.mkdtemp()), log=lambda _m: None, workers=w)
    for r in range(4):
        x, y = load_rep(seq / f"rep_{r:03d}.npz"), load_rep(par / f"rep_{r:03d}.npz")
        assert x["runtime"] == y["runtime"] and x["spec"] == y["spec"]
        np.testing.assert_array_equal(x["p"], y["p"])


def test_workers_clamped_not_rejected():
    import tempfile
    cpus = os.cpu_count() or 1
    msgs = []
    assert resolve_workers(cpus + 5, msgs.append) == cpus
    assert any("warning" in m for m in msgs)
    assert resolve_workers(1, msgs.append) == 1
    for bad in (0, -2, 1.5, True):
        try:
            resolve_workers(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted workers = {bad!r}")
    msgs.clear()
    entry = {"name": "clamp", "type": "sweep", "n": [10], "K_values": ["n"], "L": 1.0, "repetitions": 2,
             "max_iterations": 5_000, "seed": 1, "workers": cpus + 3}  # YAML key above the CPU count
    out = run_experiment(entry, Path(tempfile.mkdtemp()), log=msgs.append)
    assert (out / "summary.csv").exists() and any("warning" in m for m in msgs)
    try:
        parse_experiment({**entry, "workers": 0})
    except ValueError:
        pass
    else:
        raise AssertionError("accepted workers: 0 in an experiment entry")


def test_progress_callback_does_not_change_run():
    calls = []
    kw = dict(n=50, K=1e9, L=1.0, comparator=binval_comparator, max_iterations=1000)
    a = run_cga(**kw, rng=np.random.default_rng(0), on_progress=calls.append, progress_every=100)
    b = run_cga(**kw, rng=np.random.default_rng(0))
    assert calls == list(range(100, 1001, 100))
    assert (a.converged, a.runtime, a.iterations) == (b.converged, b.runtime, b.iterations)


class _Stop(Exception):
    pass


def test_interrupt_terminates_workers():
    import math
    import multiprocessing
    import time
    from cga.experiments import _iter_results
    if (os.cpu_count() or 1) < 2:
        print("  (skipped: fewer than 2 CPUs)")
        return
    jobs = [(r, InstanceSpec(200, 5 * math.log(200), 1.0, "binval", 50_000_000, r)) for r in range(4)]
    seen = []

    def progress(done, total, msg):  # stop at the first heartbeat, like a Streamlit rerun
        seen.append(msg)
        if "in progress" in msg:
            raise _Stop

    try:
        list(_iter_results(jobs, 2, progress, str))
    except _Stop:
        pass
    deadline = time.time() + 5
    while multiprocessing.active_children() and time.time() < deadline:
        time.sleep(0.1)
    assert not multiprocessing.active_children(), "worker processes still running after interruption"
    assert any("in progress" in m for m in seen)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
