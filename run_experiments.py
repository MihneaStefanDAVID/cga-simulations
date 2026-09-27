#!/usr/bin/env python3
"""Run the experiments listed in a YAML config and plot the results.

    python run_experiments.py                     # every experiment in experiments.yaml
    python run_experiments.py example_small       # only the experiment with that name
    python run_experiments.py --plot-only         # replot from saved data, no simulation
    python run_experiments.py NAME --fresh        # discard saved results for NAME and rerun
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from cga.experiments import parse_experiment, run_experiment

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="run only the experiment with this name")
    ap.add_argument("--config", type=Path, default=HERE / "experiments.yaml")
    ap.add_argument("--results", type=Path, default=HERE / "results", help="output root directory")
    ap.add_argument("--fresh", action="store_true", help="ignore and overwrite previously saved results")
    ap.add_argument("--plot-only", action="store_true", help="only replot from saved results")
    args = ap.parse_args(argv)

    with args.config.open() as fh:
        doc = yaml.safe_load(fh) or {}
    entries = doc.get("experiments", [])
    if not isinstance(entries, list):
        ap.error(f"{args.config}: 'experiments' must be a list")

    names = [e.get("name") for e in entries]
    dupes = {x for x in names if names.count(x) > 1}
    if dupes:
        ap.error(f"{args.config}: duplicate experiment names {sorted(dupes)}")
    for e in entries:  # validate the whole file before running anything
        parse_experiment(e)

    if args.name is not None:
        entries = [e for e in entries if e.get("name") == args.name]
        if not entries:
            ap.error(f"no experiment named '{args.name}' in {args.config}; available: {names}")

    for entry in entries:
        run_experiment(entry, args.results, fresh=args.fresh, plot_only=args.plot_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
