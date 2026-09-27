"""Regenerate the figures used in README.md.

    python docs/make_figures.py

Runs small illustrative experiments into docs/_build/ (not tracked by git) and copies the plots
to docs/images/. The parameter values are for illustration only.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from cga.experiments import run_experiment  # noqa: E402
from cga.plotting import GRID, REFERENCE, SERIES, TEXT, TEXT_MUTED  # noqa: E402

IMAGES = ROOT / "docs" / "images"
BUILD = ROOT / "docs" / "_build"

EXAMPLES = [
    {
        "name": "readme_trajectory", "type": "trajectory", "n": 200, "K": "8*log(n)", "L": 2.0,
        "repetitions": 5, "max_iterations": 1_000_000, "seed": 1,
    },
    {
        "name": "readme_sweep", "type": "sweep", "n": [100],
        "K_values": ["2*log(n)", "5*log(n)", "10*log(n)", "sqrt(n*log(n))", "0.3*n", "n"],
        "L": 2.0, "repetitions": 10, "max_iterations": 30_000, "seed": 1,
    },
]


def iteration_figure(path: Path) -> None:
    """Diagram of one cGA iteration on BinVal with n = 10."""
    X = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1]
    Y = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]
    n = len(X)
    h = next(i for i in range(n) if X[i] != Y[i])
    W, L = (X, Y) if X[h] else (Y, X)
    delta = [w - l for w, l in zip(W, L)]

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(-2.6, n + 0.2)
    ax.set_ylim(-0.2, 5.4)
    ax.invert_yaxis()
    ax.axis("off")

    blue, orange = SERIES[0], SERIES[1]
    rows = [("X", X), ("Y", Y)]
    for r, (label, bits) in enumerate(rows):
        y = 1 + r
        ax.text(-0.25, y + 0.4, label + ("  (winner)" if bits is W else "  (loser)"), ha="right",
                va="center", fontsize=11, color=TEXT)
        for i, b in enumerate(bits):
            face = blue if b else "#f1f0ec"
            ax.add_patch(Rectangle((i + 0.05, y), 0.9, 0.8, facecolor=face, edgecolor="white", lw=2))
            ax.text(i + 0.5, y + 0.4, str(b), ha="center", va="center", fontsize=12,
                    color="white" if b else TEXT_MUTED, fontweight="bold")

    for i in range(n):
        ax.text(i + 0.5, 0.7, f"{i + 1}", ha="center", va="center", fontsize=9, color=TEXT_MUTED)
    ax.text(-0.25, 0.7, "bit i", ha="right", va="center", fontsize=9, color=TEXT_MUTED)
    ax.annotate("", xy=(n, 0.25), xytext=(0, 0.25),
                arrowprops=dict(arrowstyle="->", color=REFERENCE, lw=1))
    ax.text(0, 0.1, "more significant", ha="left", va="bottom", fontsize=8.5, color=TEXT_MUTED)
    ax.text(n, 0.1, "less significant", ha="right", va="bottom", fontsize=8.5, color=TEXT_MUTED)

    # First differing bit
    ax.add_patch(FancyBboxPatch((h + 0.02, 0.95), 0.96, 1.9, boxstyle="round,pad=0.02",
                                fill=False, edgecolor=orange, lw=2.2))
    ax.text(h + 0.5, 3.15, f"h = {h + 1}", ha="center", va="center", fontsize=10, color=orange,
            fontweight="bold")

    # Update row
    y = 3.6
    ax.text(-0.25, y + 0.4, "change of $p_i$", ha="right", va="center", fontsize=11, color=TEXT)
    for i, d in enumerate(delta):
        txt, col = ("+1/K", blue) if d > 0 else ("−1/K", orange) if d < 0 else ("0", GRID)
        ax.add_patch(Rectangle((i + 0.05, y), 0.9, 0.8, facecolor="white", edgecolor=GRID, lw=1))
        ax.text(i + 0.5, y + 0.4, txt, ha="center", va="center", fontsize=10,
                color=col if d else REFERENCE, fontweight="bold" if d else "normal")

    ax.text(-2.6, 4.95,
            f"X and Y first differ at bit h = {h + 1}, where X has the 1, so X wins "
            "(it has the larger BinVal value).\n"
            "Every bit where the two differ moves 1/K toward the winner. This includes bits after h, "
            f"where the winner happens to have a 0 (bits {', '.join(str(i + 1) for i in range(n) if delta[i] < 0)}).",
            ha="left", va="top", fontsize=9.5, color=TEXT_MUTED)
    ax.set_title("One cGA iteration on BinVal (n = 10)", loc="left", fontsize=12, color=TEXT, pad=4)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    iteration_figure(IMAGES / "cga_iteration.png")
    for entry in EXAMPLES:
        run_experiment(entry, BUILD)
    copies = {
        BUILD / "readme_trajectory" / "heatmap.png": "example_heatmap.png",
        BUILD / "readme_trajectory" / "trajectories.png": "example_trajectories.png",
        BUILD / "readme_sweep" / "runtime_vs_K_n100.png": "example_sweep.png",
    }
    for src, dst in copies.items():
        shutil.copy(src, IMAGES / dst)
        print(f"copied {dst}")


if __name__ == "__main__":
    main()
