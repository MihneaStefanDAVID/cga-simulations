"""Experiment types: "trajectory" and "sweep".

The unit of work is ``run_instance(spec) -> RunResult``: one cGA run fully
determined by an ``InstanceSpec`` (n, K, L, fitness, budget, seed). It has no
side effects and both arguments and result are picklable, so it can be
dispatched to a process pool later without restructuring.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from . import plotting
from .comparators import make_comparator
from .expressions import eval_expr
from .simulator import RunResult, run_cga

# Optional hooks so that a UI can follow progress; the CLI just uses print.
LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int, str], None]  # (done, total, message)


def _no_progress(done: int, total: int, message: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# Single instance
# --------------------------------------------------------------------------- #


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
    """Run one cGA instance. Deterministic given spec (seeded), side-effect free."""
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


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #

_COMMON_REQUIRED = {"name", "type", "L", "repetitions", "max_iterations", "seed"}
_COMMON_OPTIONAL = {"fitness"}


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
        ns = entry["n"] if isinstance(entry["n"], list) else [entry["n"]]
        for n in ns:
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError(f"experiment '{entry['name']}': invalid n {n!r}")
        K_exprs = [str(k) for k in entry["K_values"]]
        if len(set(K_exprs)) != len(K_exprs):
            raise ValueError(f"experiment '{entry['name']}': duplicate entries in K_values")
        for n in ns:  # fail early on bad expressions
            for k in K_exprs:
                if not eval_expr(k, n) > 0:
                    raise ValueError(f"experiment '{entry['name']}': K = {k} is not positive for n = {n}")
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


def default_track_indices(n: int) -> list[int]:
    return [1, 2, 3, 5, 10, n // 4, n // 2, n]


def parse_experiment(entry: dict):
    kind = entry.get("type")
    if kind == "trajectory":
        return TrajectoryConfig.from_entry(entry)
    if kind == "sweep":
        return SweepConfig.from_entry(entry)
    raise ValueError(f"experiment '{entry.get('name')}': unknown type {kind!r} (use 'trajectory' or 'sweep')")


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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    reps = []
    for r in range(cfg.repetitions):
        progress(r, cfg.repetitions, f"repetition {r + 1}/{cfg.repetitions}")
        spec = _trajectory_spec(cfg, r)
        path = out_dir / f"rep_{r:03d}.npz"
        if path.exists() and not fresh:
            rep = load_rep(path)
            if rep["spec"] == asdict(spec):
                log(f"  rep {r}: loaded from {path.name}")
                reps.append(rep)
                continue
            if plot_only:
                raise RuntimeError(f"{path} was produced with different parameters; rerun without --plot-only")
        elif plot_only:
            raise RuntimeError(f"{path} does not exist; rerun without --plot-only")
        t0 = time.perf_counter()
        res = run_instance(spec)
        wall = time.perf_counter() - t0
        _save_rep(path, spec, res, wall)
        status = f"T = {res.runtime}" if res.converged else "not converged"
        log(f"  rep {r}: {status} ({wall:.1f}s)")
        reps.append(load_rep(path))
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
# Sweep experiments
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


def sweep_tasks(cfg: SweepConfig) -> list[SweepTask]:
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


def run_sweep(
    cfg: SweepConfig,
    out_dir: Path,
    fresh: bool = False,
    plot_only: bool = False,
    log: LogFn = print,
    progress: ProgressFn = _no_progress,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.csv"
    if fresh and raw_path.exists() and not plot_only:
        raw_path.unlink()

    tasks = sweep_tasks(cfg)
    done = {_row_key(row) for row in _load_raw(raw_path).itertuples(index=False)}
    todo = [t for t in tasks if t.key() not in done]
    if plot_only and todo:
        raise RuntimeError(f"{len(todo)} instances of '{cfg.name}' have no results yet; rerun without --plot-only")
    log(f"  {len(tasks)} instances, {len(tasks) - len(todo)} already done, {len(todo)} to run")

    new_file = not raw_path.exists()
    with raw_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_COLUMNS)
        if new_file:
            writer.writeheader()
        for i, task in enumerate(todo, 1):
            progress(i - 1, len(todo), f"n={task.spec.n}, K={task.K_expr}, rep {task.repetition}")
            t0 = time.perf_counter()
            res = run_instance(task.spec)
            wall = time.perf_counter() - t0
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
    plotting.plot_sweep(cfg, summary, out_dir)


def sweep_summary(cfg: SweepConfig, raw: pd.DataFrame) -> pd.DataFrame:
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
) -> Path:
    """Run one config entry; returns the directory holding its results and plots."""
    cfg = parse_experiment(entry)
    out_dir = results_root / cfg.name
    log(f"=== {cfg.name} ({entry['type']}) -> {out_dir}")
    runner = run_trajectory if isinstance(cfg, TrajectoryConfig) else run_sweep
    runner(cfg, out_dir, fresh=fresh, plot_only=plot_only, log=log, progress=progress)
    return out_dir
