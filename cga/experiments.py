"""Experiment types: "trajectory", "sweep" (runtime vs K) and "scaling" (runtime vs n).

The unit of work is ``run_instance(spec) -> RunResult`` (in cga/instance.py): one
cGA run fully determined by an ``InstanceSpec`` (n, K, L, fitness, budget, seed).
It has no side effects and is picklable; with ``workers > 1`` the instances of an
experiment run on a pool of separate OS processes.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yaml

from . import kernel, plotting
from .analysis import scaling_fits
from .expressions import eval_expr
from .instance import SLOT_FIELDS, InstanceSpec, init_worker, run_instance, timed_run  # noqa: F401  (re-exported)
from .simulator import RunResult

# Optional hooks so that a UI can follow progress; the CLI just uses print.
LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]  # (done, total, message)


def _no_progress(done: int, total: int, message: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# Running instances, sequentially or on a pool of worker processes
# --------------------------------------------------------------------------- #


def default_workers() -> int:
    """Default number of worker processes for the app and the CLI: all CPUs but one."""
    return max(1, (os.cpu_count() or 2) - 1)


def resolve_workers(requested: int, log: LogFn = print) -> int:
    """Validate a worker count; values above os.cpu_count() are clamped with a warning."""
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError(f"workers must be a positive integer, got {requested!r}")
    cpus = os.cpu_count() or 1
    if requested > cpus:
        log(f"  warning: workers = {requested} exceeds the {cpus} available CPUs; using {cpus}")
        return cpus
    return requested


HEARTBEAT_SECONDS = 2.0
# Shared progress objects of interrupted pools: kept referenced for a while, because worker processes
# that were still starting up need them to exist (their semaphore is unlinked on garbage collection).
_KEEPALIVE: list = []


def _fmt_iters(t: float) -> str:
    return f"{t / 1e6:.1f}M" if t >= 1e6 else f"{t / 1e3:.0f}k" if t >= 1e3 else f"{int(t)}"


def _live_status(shared, workers: int) -> str:
    """Summary of the runs in progress, read from the workers' shared progress slots."""
    runs = [(shared[i * SLOT_FIELDS], shared[i * SLOT_FIELDS + 1], shared[i * SLOT_FIELDS + 2])
            for i in range(workers)]
    runs = [r for r in runs if r[0] > 0]
    if not runs:
        return "starting worker processes"
    ns = sorted({int(n) for n, _, _ in runs})
    lo, hi = min(t for _, t, _ in runs), max(t for _, t, _ in runs)
    budget = max(m for _, _, m in runs)
    n_txt = f"n = {ns[0]}" if len(ns) == 1 else f"n = {ns[0]}–{ns[-1]}"
    it_txt = _fmt_iters(lo) if lo == hi else f"{_fmt_iters(lo)}–{_fmt_iters(hi)}"
    pct = f"{100 * lo / budget:.0f}%" if lo == hi else f"{100 * lo / budget:.0f}–{100 * hi / budget:.0f}%"
    return f"{len(runs)} runs in progress ({n_txt}): {it_txt} of {_fmt_iters(budget)} iterations ({pct} of budget)"


def _iter_results(
    jobs: Sequence[tuple[Any, InstanceSpec]],
    workers: int,
    progress: ProgressFn,
    describe: Callable[[Any], str],
) -> Iterator[tuple[Any, RunResult, float]]:
    """Run (tag, spec) jobs and yield (tag, result, wall_seconds) as each one finishes.

    workers == 1 runs in order in this process, exactly as before. workers > 1 uses a pool of
    separate OS processes (real parallelism; the pure-Python loop is GIL-bound, so threads
    would not help) and yields in completion order. The caller writes each result to disk as
    soon as it is yielded, so an interrupted experiment keeps every finished instance.

    progress() also fires while runs are still going (iterations done so far), so long runs
    never look frozen: in parallel mode every HEARTBEAT_SECONDS, in sequential mode every few
    hundred thousand iterations.
    """
    total = len(jobs)
    if workers <= 1:
        for i, (tag, spec) in enumerate(jobs):
            label = describe(tag)
            progress(i, total, f"running {label}")

            def live(t: int, i=i, label=label, budget=spec.max_iterations) -> None:
                progress(i, total, f"running {label}: {_fmt_iters(t)} of {_fmt_iters(budget)} iterations "
                                   f"({100 * t / budget:.0f}% of budget)")
            yield (tag, *timed_run(spec, live))
        return
    kernel.available()  # build the C++ kernel once here, instead of in every worker at the same time
    ctx = multiprocessing.get_context("spawn")  # safe inside a multithreaded host such as the Streamlit server
    shared = ctx.RawArray("d", SLOT_FIELDS * workers)  # live (n, iterations, budget); one writer per slot
    counter = ctx.Value("i", 0)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker,
                               initargs=(shared, counter))
    # Jobs are submitted in their natural order (small n first), so finished results start arriving
    # quickly and partial results are useful if the experiment is interrupted.
    ordered = list(jobs)
    completed = False
    try:
        futures = {pool.submit(timed_run, spec): tag for tag, spec in ordered}
        progress(0, total, f"started {total} runs on {workers} worker processes")
        pending, done = set(futures), 0
        while pending:
            finished, pending = wait(pending, timeout=HEARTBEAT_SECONDS, return_when=FIRST_COMPLETED)
            for fut in finished:
                tag = futures[fut]
                res, wall = fut.result()
                done += 1
                progress(done, total, f"finished {describe(tag)}")
                yield tag, res, wall
            if not finished:
                progress(done, total, _live_status(shared, workers))
        completed = True
    finally:
        if completed:
            pool.shutdown(wait=True)  # every run is done, so this only waits for the workers to exit
        else:
            # Interrupted (Ctrl-C, Streamlit rerun): drop queued runs, and stop the running ones. Their
            # results could no longer be collected, and a single long run can take many minutes.
            running = list((getattr(pool, "_processes", None) or {}).values())  # shutdown() clears this
            pool.shutdown(wait=False, cancel_futures=True)
            for proc in running:
                proc.terminate()
            _KEEPALIVE.append((shared, counter))
            del _KEEPALIVE[:-4]


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #

_COMMON_REQUIRED = {"name", "type", "L", "repetitions", "max_iterations", "seed"}
# description: free text; workers: number of parallel processes. Neither affects the results.
_COMMON_OPTIONAL = {"fitness", "description", "workers"}


def _check_keys(entry: dict, required: set, optional: set) -> None:
    name = entry.get("name", "<unnamed>")
    missing = required - entry.keys()
    unknown = entry.keys() - required - optional
    if missing:
        raise ValueError(f"experiment '{name}': missing keys {sorted(missing)}")
    if unknown:
        raise ValueError(f"experiment '{name}': unknown keys {sorted(unknown)}")


def _positive_int(entry: dict, key: str) -> int:
    value = entry[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"experiment '{entry['name']}': {key} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class TrajectoryConfig:
    name: str
    n: int
    K_expr: str
    K: float
    L: float
    repetitions: int
    max_iterations: int
    seed: int
    fitness: str = "binval"
    track_indices: tuple[int, ...] = ()  # 1-based bit indices
    heatmap_repetition: int = 0

    @staticmethod
    def from_entry(entry: dict) -> "TrajectoryConfig":
        _check_keys(
            entry,
            _COMMON_REQUIRED | {"n", "K"},
            _COMMON_OPTIONAL | {"track_indices", "heatmap_repetition"},
        )
        n = _positive_int(entry, "n")
        reps = _positive_int(entry, "repetitions")
        indices = entry.get("track_indices") or default_track_indices(n)
        indices = tuple(sorted({min(max(int(i), 1), n) for i in indices}))
        heat_rep = int(entry.get("heatmap_repetition", 0))
        if not 0 <= heat_rep < reps:
            raise ValueError(f"experiment '{entry['name']}': heatmap_repetition must be in [0, {reps})")
        return TrajectoryConfig(
            name=entry["name"],
            n=n,
            K_expr=str(entry["K"]),
            K=eval_expr(entry["K"], n),
            L=float(entry["L"]),
            repetitions=reps,
            max_iterations=_positive_int(entry, "max_iterations"),
            seed=int(entry["seed"]),
            fitness=entry.get("fitness", "binval"),
            track_indices=indices,
            heatmap_repetition=heat_rep,
        )


def geometric_ns(start: int, stop: int, factor: Optional[float] = None, count: Optional[int] = None) -> list[int]:
    """Integers from start to stop, spaced geometrically by `factor` or as `count` points."""
    if factor is not None:
        vals, v = [], float(start)
        while v <= stop * (1 + 1e-9):
            vals.append(int(round(v)))
            v *= factor
    else:
        vals = [int(round(v)) for v in np.geomspace(start, stop, count)]
    return sorted(set(vals))


def _parse_ns(entry: dict) -> list[int]:
    """`n` as a number, a list, or a geometric range {from, to, factor} / {from, to, count}."""
    name, raw = entry["name"], entry["n"]
    if isinstance(raw, dict):
        unknown = raw.keys() - {"from", "to", "factor", "count"}
        if unknown or not {"from", "to"} <= raw.keys() or (("factor" in raw) == ("count" in raw)):
            raise ValueError(f"experiment '{name}': n as a range needs 'from', 'to' and exactly one of "
                             f"'factor' or 'count', e.g. {{from: 100, to: 3200, factor: 2}}")
        start, stop = raw["from"], raw["to"]
        if not (isinstance(start, int) and isinstance(stop, int) and 1 <= start <= stop):
            raise ValueError(f"experiment '{name}': n range needs integers 1 <= from <= to")
        if "factor" in raw and not float(raw["factor"]) > 1:
            raise ValueError(f"experiment '{name}': n range factor must be > 1")
        if "count" in raw and not (isinstance(raw["count"], int) and raw["count"] >= 2):
            raise ValueError(f"experiment '{name}': n range count must be an integer >= 2")
        return geometric_ns(start, stop, float(raw["factor"]) if "factor" in raw else None, raw.get("count"))
    ns = raw if isinstance(raw, list) else [raw]
    for n in ns:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"experiment '{name}': invalid n {n!r}")
    if len(set(ns)) != len(ns):
        raise ValueError(f"experiment '{name}': duplicate n values")
    return list(ns)


def _parse_K_exprs(entry: dict, ns: list[int]) -> list[str]:
    raw = entry["K_values"]
    K_exprs = [str(k) for k in (raw if isinstance(raw, list) else [raw])]
    if not K_exprs:
        raise ValueError(f"experiment '{entry['name']}': K_values is empty")
    if len(set(K_exprs)) != len(K_exprs):
        raise ValueError(f"experiment '{entry['name']}': duplicate entries in K_values")
    for n in ns:  # fail early on bad expressions
        for k in K_exprs:
            if not eval_expr(k, n) > 0:
                raise ValueError(f"experiment '{entry['name']}': K = {k} is not positive for n = {n}")
    return K_exprs


@dataclass(frozen=True)
class SweepConfig:
    name: str
    ns: tuple[int, ...]
    K_exprs: tuple[str, ...]
    L: float
    repetitions: int
    max_iterations: int
    seed: int
    fitness: str = "binval"

    @staticmethod
    def from_entry(entry: dict) -> "SweepConfig":
        _check_keys(entry, _COMMON_REQUIRED | {"n", "K_values"}, _COMMON_OPTIONAL)
        ns = _parse_ns(entry)
        K_exprs = _parse_K_exprs(entry, ns)
        return SweepConfig(
            name=entry["name"],
            ns=tuple(ns),
            K_exprs=tuple(K_exprs),
            L=float(entry["L"]),
            repetitions=_positive_int(entry, "repetitions"),
            max_iterations=_positive_int(entry, "max_iterations"),
            seed=int(entry["seed"]),
            fitness=entry.get("fitness", "binval"),
        )


@dataclass(frozen=True)
class ScalingConfig:
    """Runtime as a function of n, for K given as formulas in n (re-evaluated at every n)."""

    name: str
    ns: tuple[int, ...]
    K_exprs: tuple[str, ...]
    L: float
    repetitions: int
    max_iterations: int
    seed: int
    fitness: str = "binval"
    normalize_by: Optional[str] = None  # f(n) for the extra plot T / f(n), e.g. "n*log(n)"

    @staticmethod
    def from_entry(entry: dict) -> "ScalingConfig":
        _check_keys(entry, _COMMON_REQUIRED | {"n", "K_values"}, _COMMON_OPTIONAL | {"normalize_by"})
        ns = _parse_ns(entry)
        if len(ns) < 2:
            raise ValueError(f"experiment '{entry['name']}': a scaling experiment needs at least 2 values of n "
                             f"(better 4 or more)")
        K_exprs = _parse_K_exprs(entry, ns)
        norm = entry.get("normalize_by")
        norm = str(norm).strip() if norm not in (None, "") else None
        if norm is not None:
            for n in ns:
                if not eval_expr(norm, n) > 0:
                    raise ValueError(f"experiment '{entry['name']}': normalize_by = {norm} is not positive "
                                     f"for n = {n}")
        return ScalingConfig(
            name=entry["name"],
            ns=tuple(sorted(ns)),
            K_exprs=tuple(K_exprs),
            L=float(entry["L"]),
            repetitions=_positive_int(entry, "repetitions"),
            max_iterations=_positive_int(entry, "max_iterations"),
            seed=int(entry["seed"]),
            fitness=entry.get("fitness", "binval"),
            normalize_by=norm,
        )


GridConfig = Union[SweepConfig, ScalingConfig]  # both run every (n, K formula, repetition)


def default_track_indices(n: int) -> list[int]:
    return [1, 2, 3, 5, 10, n // 4, n // 2, n]


def parse_experiment(entry: dict):
    kind = entry.get("type")
    if "workers" in entry:
        w = entry["workers"]
        if isinstance(w, bool) or not isinstance(w, int) or w < 1:
            raise ValueError(f"experiment '{entry.get('name')}': workers must be a positive integer, got {w!r}")
    if kind == "trajectory":
        return TrajectoryConfig.from_entry(entry)
    if kind == "sweep":
        return SweepConfig.from_entry(entry)
    if kind == "scaling":
        return ScalingConfig.from_entry(entry)
    raise ValueError(f"experiment '{entry.get('name')}': unknown type {kind!r} "
                     f"(use 'trajectory', 'sweep' or 'scaling')")


# --------------------------------------------------------------------------- #
# Trajectory experiments
# --------------------------------------------------------------------------- #


def _trajectory_spec(cfg: TrajectoryConfig, rep: int) -> InstanceSpec:
    return InstanceSpec(
        n=cfg.n,
        K=cfg.K,
        L=cfg.L,
        fitness=cfg.fitness,
        max_iterations=cfg.max_iterations,
        seed=cfg.seed + rep,
        log_checkpoints=True,
    )


def _save_rep(path: Path, spec: InstanceSpec, res: RunResult, wall: float) -> None:
    np.savez_compressed(
        path,
        times=res.checkpoint_times,
        p=res.checkpoint_p,
        converged=res.converged,
        runtime=-1 if res.runtime is None else res.runtime,
        iterations=res.iterations,
        wall_seconds=wall,
        spec=json.dumps(asdict(spec)),
    )


def load_rep(path: Path) -> dict:
    with np.load(path) as data:
        return {
            "times": data["times"],
            "p": data["p"],
            "converged": bool(data["converged"]),
            "runtime": None if int(data["runtime"]) < 0 else int(data["runtime"]),
            "iterations": int(data["iterations"]),
            "wall_seconds": float(data["wall_seconds"]),
            "spec": json.loads(str(data["spec"])),
        }


def run_trajectory(
    cfg: TrajectoryConfig,
    out_dir: Path,
    fresh: bool = False,
    plot_only: bool = False,
    log: LogFn = print,
    progress: ProgressFn = _no_progress,
    workers: int = 1,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_rep: dict[int, dict] = {}
    todo: list[tuple[int, InstanceSpec]] = []
    for r in range(cfg.repetitions):
        spec = _trajectory_spec(cfg, r)
        path = out_dir / f"rep_{r:03d}.npz"
        if path.exists() and not fresh:
            rep = load_rep(path)
            if rep["spec"] == asdict(spec):
                log(f"  rep {r}: loaded from {path.name}")
                by_rep[r] = rep
                continue
            if plot_only:
                raise RuntimeError(f"{path} was produced with different parameters; rerun without --plot-only")
        elif plot_only:
            raise RuntimeError(f"{path} does not exist; rerun without --plot-only")
        todo.append((r, spec))

    specs = dict(todo)
    for r, res, wall in _iter_results(todo, workers, progress,
                                      lambda r: f"repetition {r + 1}/{cfg.repetitions}"):
        path = out_dir / f"rep_{r:03d}.npz"
        _save_rep(path, specs[r], res, wall)  # written as soon as it finishes
        status = f"T = {res.runtime}" if res.converged else "not converged"
        log(f"  rep {r}: {status} ({wall:.1f}s)")
        by_rep[r] = load_rep(path)
    reps = [by_rep[r] for r in range(cfg.repetitions)]
    progress(cfg.repetitions, cfg.repetitions, "plotting")

    summary = trajectory_summary(cfg, reps)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    text = format_trajectory_summary(summary)
    (out_dir / "summary.txt").write_text(text)
    log(text)
    plotting.plot_trajectory(cfg, reps, out_dir)


def trajectory_summary(cfg: TrajectoryConfig, reps: list[dict]) -> dict:
    runtimes = [r["runtime"] for r in reps if r["converged"]]
    summary: dict[str, Any] = {
        "name": cfg.name,
        "n": cfg.n,
        "K_expr": cfg.K_expr,
        "K": cfg.K,
        "L": cfg.L,
        "fitness": cfg.fitness,
        "max_iterations": cfg.max_iterations,
        "repetitions": len(reps),
        "converged": len(runtimes),
        "runtimes": [r["runtime"] for r in reps],  # None = not converged
    }
    if runtimes:
        summary.update(
            runtime_min=int(np.min(runtimes)),
            runtime_median=float(np.median(runtimes)),
            runtime_max=int(np.max(runtimes)),
        )
    return summary


def format_trajectory_summary(s: dict) -> str:
    lines = [
        f"Experiment {s['name']} (trajectory, fitness={s['fitness']})",
        f"  n = {s['n']}, K = {s['K_expr']} = {s['K']:.4g}, L = {s['L']}, budget = {s['max_iterations']:,}",
        f"  converged: {s['converged']} / {s['repetitions']}",
    ]
    if s["converged"]:
        lines.append(
            f"  runtime over converged runs: min {s['runtime_min']:,}, "
            f"median {s['runtime_median']:,.1f}, max {s['runtime_max']:,}"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Sweep and scaling experiments: a grid of (n, K formula, repetition) runs
# --------------------------------------------------------------------------- #

RAW_COLUMNS = [
    "n", "K_expr", "K", "L", "fitness", "max_iterations", "seed", "repetition",
    "converged", "runtime", "iterations", "wall_seconds",
]
# Columns identifying one instance; used to skip already-finished work on resume.
KEY_COLUMNS = ["n", "K_expr", "L", "fitness", "max_iterations", "seed", "repetition"]


@dataclass(frozen=True)
class SweepTask:
    K_expr: str
    repetition: int
    spec: InstanceSpec = field(compare=False)

    def key(self) -> tuple:
        s = self.spec
        return (s.n, self.K_expr, s.L, s.fitness, s.max_iterations, s.seed, self.repetition)


def sweep_tasks(cfg: GridConfig) -> list[SweepTask]:
    tasks = []
    for n in cfg.ns:
        for k_expr in cfg.K_exprs:
            K = eval_expr(k_expr, n)
            for r in range(cfg.repetitions):
                spec = InstanceSpec(n, K, cfg.L, cfg.fitness, cfg.max_iterations, cfg.seed + r)
                tasks.append(SweepTask(k_expr, r, spec))
    return tasks


def _load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.read_csv(path, dtype={"K_expr": str, "fitness": str})


def _row_key(row) -> tuple:
    return (
        int(row.n), str(row.K_expr), float(row.L), str(row.fitness),
        int(row.max_iterations), int(row.seed), int(row.repetition),
    )


def _run_grid(
    cfg: GridConfig,
    out_dir: Path,
    fresh: bool,
    plot_only: bool,
    log: LogFn,
    progress: ProgressFn,
    workers: int = 1,
) -> pd.DataFrame:
    """Run (or resume) every instance of the grid; write raw_results.csv and summary.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.csv"
    if fresh and raw_path.exists() and not plot_only:
        raw_path.unlink()

    tasks = sweep_tasks(cfg)
    done = {_row_key(row) for row in _load_raw(raw_path).itertuples(index=False)}
    todo = [t for t in tasks if t.key() not in done]
    if plot_only and todo:
        raise RuntimeError(f"{len(todo)} instances of '{cfg.name}' have no results yet; rerun without --plot-only")
    log(f"  {len(tasks)} instances, {len(tasks) - len(todo)} already done, {len(todo)} to run"
        + (f" on {workers} worker processes" if workers > 1 and todo else ""))

    new_file = not raw_path.exists()
    with raw_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_COLUMNS)
        if new_file:
            writer.writeheader()
            fh.flush()
        jobs = [(task, task.spec) for task in todo]
        describe = lambda t: f"n={t.spec.n}, K={t.K_expr}, rep {t.repetition}"  # noqa: E731
        for i, (task, res, wall) in enumerate(_iter_results(jobs, workers, progress, describe), 1):
            s = task.spec
            writer.writerow({
                "n": s.n, "K_expr": task.K_expr, "K": repr(s.K), "L": repr(s.L),
                "fitness": s.fitness, "max_iterations": s.max_iterations, "seed": s.seed,
                "repetition": task.repetition, "converged": int(res.converged),
                "runtime": "" if res.runtime is None else res.runtime,
                "iterations": res.iterations, "wall_seconds": f"{wall:.4f}",
            })
            fh.flush()  # every finished instance survives an interruption
            status = f"T = {res.runtime}" if res.converged else "not converged"
            log(f"  [{i}/{len(todo)}] n={s.n} K={task.K_expr} (={s.K:.4g}) rep {task.repetition}: "
                  f"{status} ({wall:.1f}s)")

    progress(len(todo), len(todo), "plotting")
    raw = _load_raw(raw_path)
    wanted = {t.key() for t in tasks}
    mask = [(_row_key(row) in wanted) for row in raw.itertuples(index=False)]
    raw = raw[mask].drop_duplicates(subset=KEY_COLUMNS, keep="last")
    summary = sweep_summary(cfg, raw)
    summary.to_csv(out_dir / "summary.csv", index=False)
    log(summary.to_string(index=False))
    return summary


def run_sweep(
    cfg: SweepConfig,
    out_dir: Path,
    fresh: bool = False,
    plot_only: bool = False,
    log: LogFn = print,
    progress: ProgressFn = _no_progress,
    workers: int = 1,
) -> None:
    summary = _run_grid(cfg, out_dir, fresh, plot_only, log, progress, workers)
    plotting.plot_sweep(cfg, summary, out_dir)


def run_scaling(
    cfg: ScalingConfig,
    out_dir: Path,
    fresh: bool = False,
    plot_only: bool = False,
    log: LogFn = print,
    progress: ProgressFn = _no_progress,
    workers: int = 1,
) -> None:
    summary = _run_grid(cfg, out_dir, fresh, plot_only, log, progress, workers)
    fits = scaling_fits(summary, cfg.K_exprs)
    fits.to_csv(out_dir / "scaling_fits.csv", index=False)
    for f in fits.itertuples(index=False):
        if np.isfinite(f.exponent_b):
            log(f"  K = {f.K_expr}: T ~ {f.prefactor:.3g} * n^{f.exponent_b:.3f}  "
                f"(R^2 = {f.r_squared:.3f}, {f.n_points_used}/{f.n_points_total} n values used)")
        else:
            log(f"  K = {f.K_expr}: no fit (fewer than 2 n values where more than half of the runs finished)")
    plotting.plot_scaling(cfg, summary, fits, out_dir)


def sweep_summary(cfg: GridConfig, raw: pd.DataFrame) -> pd.DataFrame:
    """Per (n, K): success rate and runtime statistics.

    runtime_* columns use successful runs only. median_censored is the median
    over all repetitions with failed runs counted as +infinity (i.e. above the
    budget). It is unbiased by censoring, but only defined (non-NaN) when more
    than half of the runs succeeded.
    """
    rows = []
    for n in cfg.ns:
        for k_expr in cfg.K_exprs:
            sub = raw[(raw["n"] == n) & (raw["K_expr"] == k_expr)]
            conv = sub["converged"].astype(bool).to_numpy()
            rt = sub["runtime"].to_numpy(dtype=float)
            ok = rt[conv]
            reps = len(sub)
            row: dict[str, Any] = {
                "n": n,
                "K_expr": k_expr,
                "K": eval_expr(k_expr, n),
                "repetitions": reps,
                "successes": int(conv.sum()),
                "success_rate": conv.mean() if reps else np.nan,
            }
            if len(ok):
                row.update(
                    runtime_min=ok.min(),
                    runtime_p10=np.percentile(ok, 10),
                    runtime_median=np.median(ok),
                    runtime_p90=np.percentile(ok, 90),
                    runtime_max=ok.max(),
                    runtime_mean=ok.mean(),
                )
            censored = np.where(conv, rt, np.inf)
            med = np.median(censored) if reps else np.nan
            row["median_censored"] = med if np.isfinite(med) else np.nan
            rows.append(row)
    cols = ["n", "K_expr", "K", "repetitions", "successes", "success_rate", "runtime_min",
            "runtime_p10", "runtime_median", "runtime_p90", "runtime_max", "runtime_mean",
            "median_censored"]
    return pd.DataFrame(rows).reindex(columns=cols)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


def run_experiment(
    entry: dict,
    results_root: Path,
    fresh: bool = False,
    plot_only: bool = False,
    log: LogFn = print,
    progress: ProgressFn = _no_progress,
    workers: Optional[int] = None,
) -> Path:
    """Run one config entry; returns the directory holding its results and plots.

    workers: number of parallel processes. None means the entry's own `workers` key, or 1
    (sequential) if it has none; an explicit value overrides the entry. Values above
    os.cpu_count() are clamped with a warning. The results do not depend on it.
    """
    cfg = parse_experiment(entry)
    out_dir = results_root / cfg.name
    log(f"=== {cfg.name} ({entry['type']}) -> {out_dir}")
    workers = resolve_workers(entry.get("workers", 1) if workers is None else workers, log)
    runner = {TrajectoryConfig: run_trajectory, SweepConfig: run_sweep, ScalingConfig: run_scaling}[type(cfg)]
    runner(cfg, out_dir, fresh=fresh, plot_only=plot_only, log=log, progress=progress, workers=workers)
    # The exact entry (including any description), so results can be traced back and rerun.
    (out_dir / "experiment.yaml").write_text(
        yaml.safe_dump({"experiments": [entry]}, sort_keys=False, allow_unicode=True))
    return out_dir
