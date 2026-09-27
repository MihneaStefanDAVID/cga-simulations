"""Interactive UI for the cGA simulator.

Start with:  streamlit run app.py      (or double-click start_ui.command on macOS)

It uses the same pipeline as run_experiments.py. Runs started here are *drafts*
(results/.drafts/) until the user saves them under a name and description; saved
experiments live in results/<name>/, like those run from the terminal.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from cga.experiments import SweepConfig, TrajectoryConfig, default_track_indices, parse_experiment, run_experiment
from cga.expressions import eval_expr
from cga.simulator import borders

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONFIG = HERE / "experiments.yaml"
TEMPLATE = HERE / "experiment_template.yaml"
DRAFTS = RESULTS / ".drafts"  # unsaved runs: DRAFTS/<hash of settings>/unsaved/
DRAFT_NAME = "unsaved"
DRAFT_MAX_AGE_DAYS = 7
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
IMPLEMENTED_FITNESS = ["binval"]  # other comparators are registered as stubs in cga/comparators.py

st.set_page_config(page_title="cGA simulator", page_icon="🧬", layout="wide")

# --------------------------------------------------------------------------- #
# Widget state: defaults live in session_state so values survive mode switches
# --------------------------------------------------------------------------- #

DEFAULTS = {
    "tr_export_name": "my_trajectory",
    "tr_n": 200,
    "tr_K": "5*log(n)",
    "tr_L": 1.0,
    "tr_reps": 5,
    "tr_maxit": 200_000,
    "tr_seed": 42,
    "tr_fitness": "binval",
    "tr_track": "",
    "tr_heat": 0,
    "sw_export_name": "my_sweep",
    "sw_ns": "50, 100",
    "sw_Ks": "2*log(n)\n5*log(n)\nsqrt(n*log(n))\nn",
    "sw_L": 1.0,
    "sw_reps": 10,
    "sw_maxit": 200_000,
    "sw_seed": 42,
    "sw_fitness": "binval",
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
    # Re-assigning keeps Streamlit from dropping the value while the widget is not shown.
    st.session_state[key] = st.session_state[key]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def us_per_iteration(n: int) -> float:
    """Rough loop cost in microseconds (measured: ~5.5 us at n=200, ~11.5 us at n=1000)."""
    return 4.0 + 0.0075 * n


def fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return "< 1 s"
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def worst_case_seconds(cfg) -> float:
    """Upper bound: every run uses its whole budget."""
    if isinstance(cfg, TrajectoryConfig):
        return cfg.repetitions * cfg.max_iterations * us_per_iteration(cfg.n) * 1e-6
    per_n = sum(us_per_iteration(n) for n in cfg.ns)
    return len(cfg.K_exprs) * cfg.repetitions * cfg.max_iterations * per_n * 1e-6


def entry_yaml(entry: dict) -> str:
    return yaml.safe_dump({"experiments": [entry]}, sort_keys=False, allow_unicode=True)


def append_to_config(entry: dict) -> str | None:
    """Append entry to experiments.yaml as text (keeps existing comments). Returns an error or None."""
    doc = yaml.safe_load(CONFIG.read_text()) if CONFIG.exists() else None
    existing = [e.get("name") for e in (doc or {}).get("experiments", [])]
    if entry["name"] in existing:
        return f"experiments.yaml already has an experiment named '{entry['name']}'. Choose another name."
    block = yaml.safe_dump([entry], sort_keys=False)
    block = "\n".join("  " + line if line else line for line in block.splitlines())
    text = CONFIG.read_text() if CONFIG.exists() else "experiments:\n"
    if not text.endswith("\n"):
        text += "\n"
    CONFIG.write_text(text + "\n" + block + "\n")
    return None


def validate(entry: dict):
    try:
        return parse_experiment(entry), None
    except Exception as exc:  # show config problems in the UI instead of crashing
        return None, str(exc)


def run_with_ui(entry: dict, results_root: Path, label: str) -> Path | None:
    """Run one experiment with a progress bar and a live log. Returns the results folder."""
    log_lines: list[str] = []
    with st.status(f"Running **{label}** …", expanded=True) as status:
        bar = st.progress(0.0, text="starting")
        log_box = st.empty()
        t0 = time.perf_counter()

        def log(msg: str) -> None:
            log_lines.extend(msg.rstrip("\n").splitlines())
            log_box.code("\n".join(log_lines[-40:]), language=None)

        def progress(done: int, total: int, message: str) -> None:
            frac = done / total if total else 1.0
            bar.progress(min(frac, 1.0), text=f"{done}/{total} done · now: {message} · "
                                              f"elapsed {fmt_duration(time.perf_counter() - t0)}")

        try:
            out_dir = run_experiment(entry, results_root, log=log, progress=progress)
        except Exception as exc:
            status.update(label=f"**{label}** failed: {exc}", state="error")
            return None
        status.update(label=f"**{label}** finished in {fmt_duration(time.perf_counter() - t0)} (not saved yet)",
                      state="complete", expanded=False)
    return out_dir


def _core(entry: dict) -> dict:
    """The settings that determine the simulation (everything except name and description)."""
    return {k: v for k, v in entry.items() if k not in ("name", "description")}


def run_draft(entry: dict, label: str) -> dict | None:
    """Run an experiment as an unsaved draft. Returns a run record for show_run(), or None on failure.

    The draft folder is keyed by the settings, so rerunning identical settings continues an
    interrupted run instead of starting over (runs are deterministic given the seed).
    """
    key = hashlib.sha1(json.dumps(_core(entry), sort_keys=True).encode()).hexdigest()[:12]
    out = run_with_ui({"name": DRAFT_NAME, **_core(entry)}, DRAFTS / key, label)
    if out is None:
        return None
    return {"id": f"{key}_{time.time_ns()}", "dir": str(out), "entry": entry, "saved": None}


def save_run(run: dict, name: str, description: str, overwrite: bool, add_to_config: bool) -> str | None:
    """Move a draft to results/<name>/ with its description. Returns an error message or None."""
    name, description = name.strip(), description.strip()
    if not NAME_RE.fullmatch(name):
        return ("Please choose a name made of letters, digits, '_', '-' or '.', starting with a letter or digit "
                "(no spaces).")
    target = RESULTS / name
    if target.exists() and not overwrite:
        return f"An experiment named **{name}** already exists. Choose another name or tick *Overwrite*."
    entry = {"name": name, **({"description": description} if description else {}), **_core(run["entry"])}
    if add_to_config:
        err = append_to_config(entry)
        if err:
            return err
    if target.exists():
        shutil.rmtree(target)
    src = Path(run["dir"])
    shutil.move(str(src), str(target))
    shutil.rmtree(src.parent, ignore_errors=True)
    # Redraw plots and summaries so they carry the real name, and store the final experiment.yaml.
    run_experiment(entry, RESULTS, plot_only=True, log=lambda _msg: None)
    run.update(saved=name, entry=entry)
    return None


def prune_old_drafts() -> None:
    if not DRAFTS.exists():
        return
    cutoff = time.time() - DRAFT_MAX_AGE_DAYS * 86400
    for d in DRAFTS.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


def read_description(out_dir: Path) -> str:
    try:
        entry = yaml.safe_load((out_dir / "experiment.yaml").read_text())["experiments"][0]
        return str(entry.get("description") or "")
    except Exception:
        return ""


def show_run(run: dict) -> None:
    """Results of a run started in the app, with the Save / Discard panel while it is a draft."""
    if run.get("discarded"):
        st.info("These results were discarded.")
        return
    if run.get("saved"):
        name = run["saved"]
        st.success(f"Saved as **{name}** in `results/{name}/`. You can reopen it any time under *Browse results*.")
        desc = run["entry"].get("description")
        if desc:
            st.markdown(f"> {desc}")
        show_results(RESULTS / name)
        return
    if not Path(run["dir"]).exists():
        st.info("These unsaved results are no longer available.")
        return

    rid = run["id"]
    st.warning(f"**Not saved yet.** These results are in a temporary folder. Save them under a name (with a "
               f"description), or discard them. Unsaved results are deleted automatically after "
               f"{DRAFT_MAX_AGE_DAYS} days.")
    with st.form(f"save_{rid}"):
        st.markdown("**💾 Save this experiment**")
        c1, c2 = st.columns([1, 2])
        name = c1.text_input("Name", value=run["entry"].get("name", ""), key=f"{rid}_name",
                             placeholder="e.g. traj_n200_K5logn_L1",
                             help="Becomes the folder results/<name>/. Letters, digits, _ - . (no spaces).")
        desc = c2.text_area("Description", value=run["entry"].get("description", ""), key=f"{rid}_desc",
                            height=100, placeholder="What is this experiment for? What did you observe?",
                            help="Free text, stored with the results in experiment.yaml and shown under Browse "
                                 "results. It does not affect the simulation.")
        c3, c4 = st.columns(2)
        overwrite = c3.checkbox("Overwrite an existing experiment with the same name", key=f"{rid}_over")
        add_cfg = c4.checkbox("Also add it to experiments.yaml", key=f"{rid}_cfg",
                              help="So that `python run_experiments.py` can rerun it from the terminal.")
        b1, b2, _ = st.columns([1, 1, 5])
        save = b1.form_submit_button("💾 Save", type="primary")
        discard = b2.form_submit_button("🗑 Discard")
    if save:
        err = save_run(run, name, desc, overwrite, add_cfg)
        if err:
            st.error(err)
        else:
            st.rerun()
    elif discard:
        shutil.rmtree(Path(run["dir"]).parent, ignore_errors=True)
        run["discarded"] = True
        st.rerun()
    show_results(Path(run["dir"]))


def show_results(out_dir: Path) -> None:
    """Display everything saved in a results folder (either experiment type)."""
    if not out_dir.exists():
        st.info("No results yet.")
        return
    if DRAFTS in out_dir.parents:
        st.caption("Files are in a temporary folder until you save them.")
    else:
        try:
            shown = out_dir.relative_to(HERE)
        except ValueError:
            shown = out_dir
        st.caption(f"Files are in `{shown}/` (inside the project folder)")
    if (out_dir / "raw_results.csv").exists():
        _show_sweep_results(out_dir)
    elif (out_dir / "summary.json").exists():
        _show_trajectory_results(out_dir)
    else:
        st.info("This folder has no finished results yet.")


def _show_trajectory_results(out_dir: Path) -> None:
    s = json.loads((out_dir / "summary.json").read_text())
    c = st.columns(5)
    c[0].metric("Converged", f"{s['converged']} / {s['repetitions']}",
                help="Runs that sampled the all-ones string within max_iterations.")
    c[1].metric("Min runtime", f"{s['runtime_min']:,}" if s["converged"] else "—")
    c[2].metric("Median runtime", f"{s['runtime_median']:,.0f}" if s["converged"] else "—")
    c[3].metric("Max runtime", f"{s['runtime_max']:,}" if s["converged"] else "—")
    c[4].metric("K", f"{s['K']:.4g}", help=f"K = {s['K_expr']}")
    runs = pd.DataFrame({
        "repetition": range(len(s["runtimes"])),
        "runtime T": [("not converged" if t is None else f"{t:,}") for t in s["runtimes"]],
    })
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Frequency heatmap** — one row per bit (bit 1 on top), one column per checkpoint. "
                    "Grey = ½ (no information), blue → 1, orange → 0. A diagonal blue front means bits are "
                    "fixed one after another from the most significant; a uniform pattern means they move together.")
        _image(out_dir / "heatmap.png")
    with right:
        st.markdown("**Runs**")
        st.dataframe(runs, hide_index=True)
    st.markdown("**Frequency trajectories** — p_i over time for selected bits. Line = mean, band = min–max over the "
                "repetitions still running at that time; dashed lines = borders l and u.")
    _image(out_dir / "trajectories.png")
    _downloads(out_dir, ["summary.json", "summary.txt", "heatmap.pdf", "trajectories.pdf"])


def _show_sweep_results(out_dir: Path) -> None:
    summary_path = out_dir / "summary.csv"
    if not summary_path.exists():
        st.info("Sweep not finished yet (no summary.csv).")
        return
    summary = pd.read_csv(summary_path)
    st.markdown("**How to read the plots.** Top: median runtime over the *successful* runs (line), 10th–90th "
                "percentile band, mean (dashed). The label under each marker is successes/repetitions — the number "
                "of runs the statistics are based on. Bottom: fraction of runs that sampled the optimum within "
                "the budget. Both axes of the top plot are logarithmic.")
    plots = sorted(out_dir.glob("runtime_vs_K_n*.png"), key=lambda p: int(p.stem.split("_n")[-1]))
    for tab, path in zip(st.tabs([f"n = {p.stem.split('_n')[-1]}" for p in plots]), plots):
        with tab:
            _image(path, max_width=900)
    st.markdown("**Summary table** (runtime columns: successful runs only; `median_censored`: median over all runs "
                "with failures counted as ∞, defined only when more than half succeeded)")
    st.dataframe(summary, hide_index=True)
    _downloads(out_dir, ["raw_results.csv", "summary.csv"] + [p.with_suffix(".pdf").name for p in plots])


def _image(path: Path, max_width: int | None = None) -> None:
    if path.exists():
        st.image(str(path), width=max_width or "stretch")


def _downloads(out_dir: Path, names: list[str]) -> None:
    files = [out_dir / n for n in names if (out_dir / n).exists()]
    if not files:
        return
    cols = st.columns(len(files))
    for col, f in zip(cols, files):
        tag = hashlib.md5(str(out_dir).encode()).hexdigest()[:8]
        col.download_button(f"⬇ {f.name}", f.read_bytes(), file_name=f.name, key=f"dl_{tag}_{f.name}")


def config_actions(core: dict, prefix: str) -> None:
    """Export the current settings as YAML (download or add to experiments.yaml) without running them here."""
    with st.expander("Export as YAML instead (e.g. to run a long experiment from the terminal)"):
        name = st.text_input("Name for the exported experiment", key=f"{prefix}_export_name").strip()
        if not NAME_RE.fullmatch(name):
            st.error("Use letters, digits, '_', '-' or '.', starting with a letter or digit (no spaces).")
            return
        entry = {"name": name, **core}
        text = entry_yaml(entry)
        st.code(text, language="yaml")
        c1, c2 = st.columns(2)
        c1.download_button("⬇ Download as YAML file", text, file_name=f"{entry['name']}.yaml",
                           key=f"{prefix}_dl_yaml")
        if c2.button("Add to experiments.yaml", key=f"{prefix}_append",
                     help="Appends this entry to the project's experiments.yaml, so "
                          "`python run_experiments.py` also runs it. Existing comments are kept."):
            err = append_to_config(entry)
            if err:
                st.error(err)
            else:
                st.success(f"Added '{entry['name']}' to experiments.yaml.")


def glossary() -> None:
    with st.expander("📖 Variables and symbols"):
        st.markdown(
            "| Symbol | Meaning |\n|---|---|\n"
            "| **n** | number of bits (problem size) |\n"
            "| **p_i** | frequency: probability that bit i of a sample is 1. Starts at ½ |\n"
            "| **K** | hypothetical population size; each update moves p_i by the step size **1/K**. "
            "Larger K = smaller, more careful steps |\n"
            "| **L** | border factor; the borders are **l = 1/(L·n)** and **u = 1 − l**. "
            "L = 1 gives the standard borders 1/n and 1 − 1/n; larger L lets p_i get closer to 0 and 1 |\n"
            "| **X, Y** | the two individuals sampled per iteration |\n"
            "| **W, L′** | winner and loser of the comparison |\n"
            "| **T** | runtime: the first iteration in which X or Y is the all-ones optimum |\n"
            "| **max_iterations** | budget per run; if T is not reached by then, the run is *not converged* |\n"
            "| **seed** | base random seed; repetition r uses seed + r, so every run is reproducible |\n"
            "| **BinVal** | fitness Σ 2^(n−i)·x_i; comparing two strings = the first bit where they differ decides |"
        )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def page_about() -> None:
    st.title("Compact Genetic Algorithm on BinVal")
    st.markdown(
        "This tool simulates the **compact Genetic Algorithm (cGA)** and plots what it does. "
        "Choose a mode in the sidebar:\n\n"
        "- **Trajectory** — one setting (n, K, L), watched over time: how do the frequencies evolve?\n"
        "- **Sweep** — runtime as a function of K, for one or more n.\n"
        "- **Load from file** — upload an experiment file (YAML) with any number of experiments and run them.\n"
        "- **Browse results** — reopen the plots and tables of any experiment run before (from here or the terminal)."
    )
    st.subheader("The algorithm")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("The state is a frequency vector $p = (p_1, \\dots, p_n)$, initially $p_i = \\tfrac12$. "
                    "In every iteration $t = 1, 2, \\dots$:")
        st.markdown(
            "1. **Sample** two strings $X, Y \\in \\{0,1\\}^n$ independently, with $\\Pr[X_i = 1] = p_i$.\n"
            "2. **Stop** if $X$ or $Y$ is the optimum $1^n$; the runtime is $T = t$.\n"
            "3. **Compare** them. Under BinVal the first position $h$ where $X$ and $Y$ differ decides: the one "
            "with a 1 there is the winner $W$, the other the loser $L'$. If $X = Y$, nothing happens this round.\n"
            "4. **Update** every bit where winner and loser differ, toward the winner:"
        )
        st.latex(r"p_i \leftarrow p_i + \tfrac{1}{K}\,(W_i - L'_i), \qquad "
                 r"\text{then clip } p_i \text{ to } \left[\tfrac{1}{Ln},\; 1 - \tfrac{1}{Ln}\right].")
    with c2:
        st.markdown("**Why BinVal is compared bit by bit**")
        st.markdown("$\\mathrm{BinVal}(x) = \\sum_i 2^{n-i} x_i$. The highest differing bit outweighs all lower "
                    "bits together ($2^{k} > 2^{k-1} + \\dots + 1$), so comparing values is the same as comparing the "
                    "strings lexicographically. The simulator never builds the (huge) number.")
        st.markdown("**Budget**: every run stops after `max_iterations` at the latest and is then recorded as "
                    "*not converged*, so nothing loops forever.")
    glossary()
    st.subheader("Where things are saved")
    st.markdown(
        "Nothing is saved automatically. After a run, the results are shown together with a **Save** panel: give the "
        "experiment a name and a description, and it is stored in `results/<name>/` inside the project folder "
        "(plots, data, and the exact configuration in `experiment.yaml`). Saved experiments appear under "
        "*Browse results*. Until then the run is a temporary draft; you can discard it, and unsaved drafts are "
        f"deleted after {DRAFT_MAX_AGE_DAYS} days. If a run is interrupted, starting it again with the same settings "
        "continues where it stopped.")


def page_trajectory() -> None:
    st.title("Trajectory — one setting over time")
    st.markdown(
        "**What happens:** the cGA runs **repetitions** times with the same n, K and L (repetition r uses "
        "seed + r). During each run the full frequency vector p is saved at t = 0, 1, 2, 4, 8, … (doubling) and once "
        "at the end. Afterwards you get a **heatmap** of all frequencies over time for one run, the **trajectories** "
        "of selected bits averaged over the runs, and a **runtime summary**."
    )
    glossary()

    st.subheader("Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("n — number of bits", key="tr_n", min_value=1, step=10)
        st.text_input("K — population size (number or expression in n)", key="tr_K",
                      help="Step size is 1/K. Examples: 50, 5*log(n), sqrt(n*log(n)), 0.3*n. log is the natural log.")
    with c2:
        st.number_input("L — border factor", key="tr_L", min_value=0.01, step=0.05, format="%.3f",
                        help="Borders l = 1/(L·n), u = 1 − l. L = 1: standard borders.")
        st.number_input("Repetitions", key="tr_reps", min_value=1, step=1)
        st.number_input("max_iterations — budget per run", key="tr_maxit", min_value=1, step=50_000,
                        help="A run that has not sampled the optimum by then is recorded as not converged.")
    with c3:
        st.number_input("Seed", key="tr_seed", step=1, help="Repetition r uses seed + r.")
        st.selectbox("Fitness function", IMPLEMENTED_FITNESS, key="tr_fitness",
                     help="Only static BinVal is implemented; OneMax and Dynamic BinVal are prepared as stubs.")
        with st.expander("Advanced"):
            st.text_input("Bits to track (comma-separated, 1-based)", key="tr_track",
                          help="Empty = default 1, 2, 3, 5, 10, n//4, n//2, n.")
            if st.session_state.tr_heat >= st.session_state.tr_reps:
                st.session_state.tr_heat = 0
            st.number_input("Repetition shown in the heatmap", key="tr_heat", min_value=0,
                            max_value=int(st.session_state.tr_reps) - 1, step=1)

    ss = st.session_state
    entry = {
        "name": DRAFT_NAME, "type": "trajectory", "fitness": ss.tr_fitness,
        "n": int(ss.tr_n), "K": ss.tr_K.strip(), "L": float(ss.tr_L), "repetitions": int(ss.tr_reps),
        "max_iterations": int(ss.tr_maxit), "seed": int(ss.tr_seed),
    }
    track_error = None
    if ss.tr_track.strip():
        try:
            entry["track_indices"] = [int(x) for x in ss.tr_track.replace(";", ",").split(",") if x.strip()]
        except ValueError:
            track_error = "Bits to track must be whole numbers separated by commas."
    if ss.tr_heat:
        entry["heatmap_repetition"] = int(ss.tr_heat)

    cfg, error = validate(entry)
    error = track_error or error

    st.subheader("Resulting values")
    if error:
        st.error(error)
    else:
        lo, hi = borders(cfg.n, cfg.L)
        m = st.columns(5)
        m[0].metric("K", f"{cfg.K:.4g}", help=f"K = {cfg.K_expr} with n = {cfg.n}")
        m[1].metric("Step size 1/K", f"{1 / cfg.K:.4g}")
        m[2].metric("Lower border l", f"{lo:.4g}", help="1/(L·n)")
        m[3].metric("Upper border u", f"{hi:.4g}", help="1 − 1/(L·n)")
        m[4].metric("Worst-case time", fmt_duration(worst_case_seconds(cfg)),
                    help="Rough estimate if every run uses its full budget. Runs that converge earlier are faster.")
        st.caption(f"Tracked bits: {', '.join(map(str, cfg.track_indices))} · "
                   f"checkpoints per run: about {int(math.log2(cfg.max_iterations)) + 2} "
                   f"(t = 0, 1, 2, 4, …, {2 ** int(math.log2(cfg.max_iterations)):,})")
        config_actions(_core(entry), "tr")

    st.subheader("Run")
    st.caption("Results are not saved automatically: after the run you can save them under a name and description. "
               "Changing any input while a run is going stops it; running the same settings again continues "
               "where it stopped.")
    if st.button("▶ Run trajectory experiment", type="primary", disabled=bool(error)):
        run = run_draft(_core(entry), "trajectory experiment")
        if run:
            ss.tr_run = run
    if ss.get("tr_run"):
        st.subheader("Results")
        show_run(ss.tr_run)


def _parse_ns(text: str) -> list[int]:
    return [int(x) for x in text.replace(";", ",").replace(" ", ",").split(",") if x.strip()]


def page_sweep() -> None:
    st.title("Sweep — runtime as a function of K")
    st.markdown(
        "**What happens:** for every combination of n and K value, the cGA runs **repetitions** independent times "
        "and records only the runtime T (or *not converged* if the budget runs out). Every finished run is appended "
        "to `raw_results.csv` right away. Afterwards you get one plot per n with the median runtime, the 10th–90th "
        "percentile band, the mean, and the **success rate** (runs that finish within the budget) — the runtime "
        "statistics only include successful runs, so always read the two panels together."
    )
    glossary()

    st.subheader("Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("n values (comma-separated)", key="sw_ns", help="One figure is produced per n.")
        st.number_input("L — border factor", key="sw_L", min_value=0.01, step=0.05, format="%.3f",
                        help="Borders l = 1/(L·n), u = 1 − l. L = 1: standard borders.")
    with c2:
        st.text_area("K values (one per line, number or expression in n)", key="sw_Ks", height=170,
                     help="Each expression is evaluated separately for every n. log is the natural log.")
    with c3:
        st.number_input("Repetitions per (n, K)", key="sw_reps", min_value=1, step=1)
        st.number_input("max_iterations — budget per run", key="sw_maxit", min_value=1, step=50_000)
        st.number_input("Seed", key="sw_seed", step=1,
                        help="Repetition r uses seed + r, for every n and K (common random numbers).")
        st.selectbox("Fitness function", IMPLEMENTED_FITNESS, key="sw_fitness")

    ss = st.session_state
    try:
        ns = _parse_ns(ss.sw_ns)
        ns_error = None if ns else "Enter at least one n."
    except ValueError:
        ns, ns_error = [], "n values must be whole numbers separated by commas."
    K_exprs = [line.strip() for line in ss.sw_Ks.splitlines() if line.strip()]
    entry = {
        "name": DRAFT_NAME, "type": "sweep", "fitness": ss.sw_fitness, "n": ns, "K_values": K_exprs,
        "L": float(ss.sw_L), "repetitions": int(ss.sw_reps), "max_iterations": int(ss.sw_maxit),
        "seed": int(ss.sw_seed),
    }
    cfg, error = validate(entry) if not ns_error else (None, ns_error)
    if not error and not K_exprs:
        error = "Enter at least one K value."

    st.subheader("Resulting values")
    if error:
        st.error(error)
    else:
        rows = []
        for k in cfg.K_exprs:
            row = {"K expression": k}
            for n in cfg.ns:
                K = eval_expr(k, n)
                row[f"K (n={n})"] = round(K, 4)
                row[f"1/K (n={n})"] = round(1 / K, 5)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Borders l = 1/(L·n), u = 1 − l:  " + "  ·  ".join(
            f"n = {n}: l = {borders(n, cfg.L)[0]:.4g}, u = {borders(n, cfg.L)[1]:.4g}" for n in cfg.ns))
        m = st.columns(3)
        m[0].metric("Runs in total", f"{len(cfg.ns) * len(cfg.K_exprs) * cfg.repetitions:,}",
                    help="n values × K values × repetitions")
        m[1].metric("Budget per run", f"{cfg.max_iterations:,}")
        m[2].metric("Worst-case time", fmt_duration(worst_case_seconds(cfg)),
                    help="Rough estimate if every run uses its full budget.")
        config_actions(_core(entry), "sw")

    st.subheader("Run")
    st.caption("Results are not saved automatically: after the run you can save them under a name and description. "
               "Changing any input while a run is going stops it; running the same settings again continues "
               "where it stopped.")
    if st.button("▶ Run sweep", type="primary", disabled=bool(error)):
        run = run_draft(_core(entry), "sweep")
        if run:
            ss.sw_run = run
    if ss.get("sw_run"):
        st.subheader("Results")
        show_run(ss.sw_run)


def _describe(cfg) -> dict:
    if isinstance(cfg, TrajectoryConfig):
        return {"name": cfg.name, "type": "trajectory", "n": str(cfg.n), "K": f"{cfg.K_expr} (= {cfg.K:.4g})",
                "L": cfg.L, "repetitions": cfg.repetitions, "max_iterations": f"{cfg.max_iterations:,}",
                "seed": cfg.seed, "runs": cfg.repetitions, "worst-case time": fmt_duration(worst_case_seconds(cfg))}
    return {"name": cfg.name, "type": "sweep", "n": ", ".join(map(str, cfg.ns)), "K": ", ".join(cfg.K_exprs),
            "L": cfg.L, "repetitions": cfg.repetitions, "max_iterations": f"{cfg.max_iterations:,}",
            "seed": cfg.seed, "runs": len(cfg.ns) * len(cfg.K_exprs) * cfg.repetitions,
            "worst-case time": fmt_duration(worst_case_seconds(cfg))}


def page_load() -> None:
    st.title("Load from file")
    st.markdown(
        "Run a whole experiment file at once. The file is YAML with a list under `experiments:`; each entry is "
        "either a `trajectory` or a `sweep` (same format as `experiments.yaml`). Start from the template — it "
        "explains every field."
    )
    c1, c2 = st.columns([1, 3])
    c1.download_button("⬇ Download template", TEMPLATE.read_bytes(), file_name="experiment_template.yaml")
    c2.caption(f"The template is also in the project folder: `{TEMPLATE.name}`")

    source = st.radio("Source", ["Upload a file", "Use the project's experiments.yaml"], horizontal=True)
    if source == "Upload a file":
        uploaded = st.file_uploader("Experiment file (.yaml / .yml)", type=["yaml", "yml"])
        if uploaded is None:
            return
        raw_text, label = uploaded.getvalue().decode("utf-8"), uploaded.name
    else:
        raw_text, label = CONFIG.read_text(), CONFIG.name

    try:
        doc = yaml.safe_load(io.StringIO(raw_text)) or {}
        entries = doc.get("experiments")
        if not isinstance(entries, list) or not entries:
            raise ValueError("the file needs a non-empty list under the key 'experiments:'")
    except Exception as exc:
        st.error(f"Could not read {label}: {exc}")
        return

    with st.expander(f"File contents — {label}"):
        st.code(raw_text, language="yaml")

    names = [e.get("name") for e in entries]
    parsed, problems = {}, []
    for e in entries:
        if names.count(e.get("name")) > 1:
            problems.append(f"duplicate name '{e.get('name')}'")
            continue
        cfg, err = validate(e)
        if err:
            problems.append(err)
        else:
            parsed[cfg.name] = (e, cfg)
    for p in problems:
        st.error(p)
    if not parsed:
        return

    st.subheader(f"{len(parsed)} valid experiment(s)")
    overview = pd.DataFrame([_describe(cfg) for _, cfg in parsed.values()])
    descriptions = [str(e.get("description") or "") for e, _ in parsed.values()]
    if any(descriptions):
        overview.insert(1, "description", descriptions)
    st.dataframe(overview, hide_index=True)
    chosen = st.multiselect("Experiments to run", list(parsed), default=list(parsed))
    total = sum(worst_case_seconds(parsed[n][1]) for n in chosen)
    st.caption(f"Worst-case total time: {fmt_duration(total)} (rough estimate, every run using its full budget). "
               "Results are not saved automatically: each one gets its own Save panel, with the name and description "
               "from the file filled in.")

    if st.button("▶ Run selected", type="primary", disabled=not chosen):
        runs = []
        for name in chosen:
            run = run_draft(parsed[name][0], name)
            if run:
                runs.append(run)
        st.session_state.lf_runs = runs
    runs = st.session_state.get("lf_runs") or []
    if runs:
        st.subheader("Results")
        labels = [("✅ " if r.get("saved") else "🗑 " if r.get("discarded") else "✏️ ") + r["entry"]["name"]
                  for r in runs]
        for tab, run in zip(st.tabs(labels), runs):
            with tab:
                show_run(run)


def page_browse() -> None:
    st.title("Browse results")
    dirs = sorted((d for d in RESULTS.glob("*") if d.is_dir() and not d.name.startswith(".")),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    if not dirs:
        st.info("No saved experiments yet. Run one and save it.")
        return
    names = [d.name for d in dirs]
    wanted = st.query_params.get("exp")
    name = st.selectbox("Experiment (most recent first)", names,
                        index=names.index(wanted) if wanted in names else 0)
    out_dir = RESULTS / name
    desc = read_description(out_dir)
    if desc:
        st.markdown(f"> {desc}")
    cfg_file = out_dir / "experiment.yaml"
    if cfg_file.exists():
        with st.expander("Configuration used"):
            st.code(cfg_file.read_text(), language="yaml")
    show_results(out_dir)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

PAGES = {
    "About the algorithm": (page_about, "what the cGA does, all symbols"),
    "Trajectory": (page_trajectory, "one (n, K, L), frequencies over time"),
    "Sweep": (page_sweep, "runtime vs K, for one or more n"),
    "Load from file": (page_load, "upload a YAML experiment file"),
    "Browse results": (page_browse, "reopen saved experiments"),
}

# Optional URL parameters: ?mode=Sweep or ?mode=Browse+results&exp=<name> (applied on first load only).
if "url_applied" not in st.session_state:
    st.session_state.url_applied = True
    prune_old_drafts()
    if st.query_params.get("mode") in PAGES:
        st.session_state.mode = st.query_params["mode"]

with st.sidebar:
    st.header("🧬 cGA simulator")
    mode = st.radio("Mode", list(PAGES), captions=[c for _, c in PAGES.values()], key="mode")
    st.divider()
    st.caption("Runs are drafts until you save them. Saved experiments go to `results/<name>/` in the project "
               "folder and appear under *Browse results*, together with experiments run from the terminal.")

PAGES[mode][0]()
