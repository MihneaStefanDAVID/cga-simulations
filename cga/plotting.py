"""Plots for trajectory and sweep experiments (matplotlib, saved as PNG + PDF)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .simulator import borders

if TYPE_CHECKING:
    import pandas as pd

    from .experiments import SweepConfig, TrajectoryConfig

# Fixed categorical order (never cycled); slot i always means the same series.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8983"

# Diverging map for frequencies: p -> 0 orange, p = 1/2 neutral gray, p -> 1 blue.
FREQ_CMAP = LinearSegmentedColormap.from_list(
    "cga_freq", ["#a8431a", "#eb6834", "#dcdbd6", "#2a78d6", "#154a8c"]
)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.edgecolor": REFERENCE,
    "axes.labelcolor": TEXT,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "text.color": TEXT,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
})


def _save(fig, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path_stem.with_suffix('.png')} (+ .pdf)")


def _is_grid_time(t: int) -> bool:
    """Doubling checkpoint grid: 0, 1, 2, 4, 8, ..."""
    return t == 0 or (t & (t - 1)) == 0


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #


def plot_trajectory(cfg: "TrajectoryConfig", reps: list[dict], out_dir: Path) -> None:
    _plot_heatmap(cfg, reps[cfg.heatmap_repetition], cfg.heatmap_repetition, out_dir)
    _plot_frequency_trajectories(cfg, reps, out_dir)


def _plot_heatmap(cfg: "TrajectoryConfig", rep: dict, rep_idx: int, out_dir: Path) -> None:
    times, P = rep["times"], rep["p"]  # P: (checkpoints, n)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(
        P.T, aspect="auto", origin="upper", interpolation="nearest",
        cmap=FREQ_CMAP, vmin=0.0, vmax=1.0,
        extent=(-0.5, len(times) - 0.5, cfg.n + 0.5, 0.5),
    )
    step = max(1, len(times) // 12)
    ticks = list(range(0, len(times), step))
    if ticks[-1] != len(times) - 1:
        ticks.append(len(times) - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{times[j]:,}" for j in ticks], rotation=45, ha="right")
    ax.set_xlabel("iteration t (checkpoints at t = 0, 1, 2, 4, …; last column = end of run)")
    ax.set_ylabel("bit index i (1 = most significant)")
    status = f"optimum sampled at T = {rep['runtime']:,}" if rep["converged"] else (
        f"not converged within {cfg.max_iterations:,}")
    ax.set_title(f"{cfg.name}: frequencies $p_{{i,t}}$, repetition {rep_idx}\n"
                 f"(n={cfg.n}, K={cfg.K:.3g}, L={cfg.L:g}; {status})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("$p_i$")
    cbar.outline.set_visible(False)
    _save(fig, out_dir / "heatmap")


def _plot_frequency_trajectories(cfg: "TrajectoryConfig", reps: list[dict], out_dir: Path) -> None:
    # Only grid checkpoints (identical across repetitions) are averaged; the
    # per-run final snapshot is used only in the heatmap. At each t the
    # statistics use the repetitions still running at t.
    grid_times = sorted({int(t) for rep in reps for t in rep["times"] if _is_grid_time(int(t))})
    cols = [np.array([int(t) for t in rep["times"]]) for rep in reps]
    idx0 = np.array(cfg.track_indices) - 1

    stacks = []  # per grid time: (reps_alive, len(indices))
    for t in grid_times:
        vals = [rep["p"][np.flatnonzero((c == t))[0], idx0] for rep, c in zip(reps, cols) if t in c]
        stacks.append(np.array(vals))
    mean = np.array([s.mean(axis=0) for s in stacks])
    lo = np.array([s.min(axis=0) for s in stacks])
    hi = np.array([s.max(axis=0) for s in stacks])
    alive = np.array([len(s) for s in stacks])

    fig, ax = plt.subplots(figsize=(8, 5))
    lo_b, hi_b = borders(cfg.n, cfg.L)
    for y in (lo_b, hi_b):
        ax.axhline(y, color=REFERENCE, lw=0.8, ls="--", zorder=1)
    ax.axhline(0.5, color=REFERENCE, lw=0.8, ls=":", zorder=1)
    x = np.array(grid_times, dtype=float)
    for j, i in enumerate(cfg.track_indices):
        color = SERIES[j % len(SERIES)] if len(cfg.track_indices) <= len(SERIES) else None
        ax.fill_between(x, lo[:, j], hi[:, j], color=color, alpha=0.10, lw=0, zorder=2)
        ax.plot(x, mean[:, j], color=color, lw=1.8, label=f"i = {i}", zorder=3)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="major")
    ax.set_xlabel("iteration t (symlog scale)")
    ax.set_ylabel("frequency $p_i$")
    ax.set_title(f"{cfg.name}: $p_{{i,t}}$ for tracked bits (n={cfg.n}, K={cfg.K:.3g}, L={cfg.L:g})")
    note = (f"line = mean, band = min–max over repetitions still running at t "
            f"({alive[0]} at t=0, {alive[-1]} at t={grid_times[-1]:,}); dashed = borders l, u")
    ax.text(0, -0.16, note, transform=ax.transAxes, fontsize=8.5, color=TEXT_MUTED)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="bit")
    _save(fig, out_dir / "trajectories")


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #


def _stagger(K: np.ndarray, ratio: float = 1.3) -> np.ndarray:
    """Mark every other K of a run of close neighbours (on the log axis) so labels don't overlap."""
    up = np.zeros(len(K), dtype=bool)
    for j in range(1, len(K)):
        if K[j] / K[j - 1] < ratio and not up[j - 1]:
            up[j] = True
    return up


def plot_sweep(cfg: "SweepConfig", summary: "pd.DataFrame", out_dir: Path) -> None:
    for n in cfg.ns:
        _plot_sweep_n(cfg, summary[summary["n"] == n].sort_values("K"), n, out_dir)


def _plot_sweep_n(cfg: "SweepConfig", s: "pd.DataFrame", n: int, out_dir: Path) -> None:
    fig, (ax, ax_s) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6.5), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )
    K = s["K"].to_numpy(float)
    ok = s["successes"].to_numpy() > 0
    reps = s["repetitions"].to_numpy()

    ax.axhline(cfg.max_iterations, color=REFERENCE, lw=0.9, ls=":", zorder=1)
    ax.annotate("budget", (1, cfg.max_iterations), xycoords=("axes fraction", "data"),
                xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom",
                fontsize=8.5, color=TEXT_MUTED)
    if ok.any():
        ax.fill_between(K[ok], s["runtime_p10"][ok], s["runtime_p90"][ok], color=SERIES[0],
                        alpha=0.15, lw=0, label="10th–90th percentile", zorder=2)
        ax.plot(K[ok], s["runtime_median"][ok], color=SERIES[0], lw=1.8, marker="o", ms=7,
                label="median", zorder=4)
        ax.plot(K[ok], s["runtime_mean"][ok], color=SERIES[1], lw=1.2, ls="--", marker="s", ms=4,
                label="mean", zorder=3)
        shift = _stagger(K)
        for k, med, succ, r, up in zip(K[ok], s["runtime_median"][ok], s["successes"][ok], reps[ok], shift[ok]):
            # Staggered points put their label above the marker instead of below.
            ax.annotate(f"{succ}/{r}", (k, med), xytext=(0, 9 if up else -9), textcoords="offset points",
                        ha="center", va="bottom" if up else "top", fontsize=8, color=TEXT_MUTED)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    else:
        ax.text(0.5, 0.5, "no repetition reached the optimum within the budget",
                transform=ax.transAxes, ha="center", color=TEXT_MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major")
    ax.set_ylabel("runtime T (iterations)")
    ax.set_title(f"{cfg.name}: cGA on {cfg.fitness}, n = {n}, L = {cfg.L:g}, "
                 f"budget {cfg.max_iterations:,}\n"
                 "runtime statistics over successful runs only (label = successes / repetitions)",
                 fontsize=10.5)

    ax_s.plot(K, s["success_rate"], color=SERIES[0], lw=1.8, marker="o", ms=6)
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.set_yticks([0, 0.5, 1])
    ax_s.grid(True, which="major")
    ax_s.set_ylabel("P(success)")
    ax_s.set_xlabel("K (log scale)")
    ax_s.set_xticks(K)
    ax_s.set_xticklabels([("\n\n\n" if up else "") + f"{e}\n({k:.3g})"
                          for e, k, up in zip(s["K_expr"], K, _stagger(K))], rotation=30,
                         ha="right", fontsize=8)
    ax_s.minorticks_off()
    ax.minorticks_off()
    ax.set_yscale("log")  # restore log minor ticks on y after minorticks_off
    _save(fig, out_dir / f"runtime_vs_K_n{n}")
