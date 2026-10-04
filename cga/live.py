"""Live runs: one long cGA run in a background process, watched while it runs.

A live run lives in results/.live/<run id>/:
  meta.json        parameters (n, K, L, fitness, seed, budget, ...) and the runner's pid
  live.bin         memory-mapped store of fixed size (see LiveStore), written by the runner and
                   read by the app; its size does not depend on how long the run takes
  checkpoint.json  exact state for resuming: iterations done, frequencies, random-number state
  stop.request     created by the app to ask the runner to stop
  runner.log       the runner's output

The runner is a separate OS process, started with `python -m cga.live run <dir>` in a new
session, so it keeps running when the browser tab or even the Streamlit app is closed. It uses
the C++ kernel, whose results are identical to the Python simulator, and it can be stopped and
resumed: the continuation is exactly the run that would have happened without the interruption.

Memory is constant because nothing is kept per iteration. The store holds:
  - the current frequency row, updated a few times per second;
  - the "cascade": a ring buffer with the most recent rows, one every k iterations, where k adapts
    so that about `rows_per_second` rows arrive per second;
  - the full history, compacted: at most H rows at times 0, d, 2d, ...; when it is full, every
    other row is dropped and d doubles, so it always spans the whole run uniformly in time;
  - a log-time history: rows at t = 1, 2^(1/8), 2^(2/8), ... (8 per doubling of t);
  - per row, four metrics: the front (leading bits at the upper border), the number of bits at
    the upper border, the number at the lower border, and the mean frequency.
Frequencies are stored as bytes (p * 255, rounded): 256 colour levels, 1 byte per bit.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import kernel
from .expressions import eval_expr
from .simulator import borders, validate_params

PROJECT = Path(__file__).resolve().parents[1]
LIVE_ROOT = PROJECT / "results" / ".live"

MAGIC = 0x43474C4956450001  # "CGLIVE" + version
STATUS = {0: "starting", 1: "running", 2: "stopped", 3: "optimum found", 4: "budget reached", 5: "error"}
STARTING, RUNNING, STOPPED, FINISHED, BUDGET, ERROR = range(6)
N_METRICS = 4  # front, bits at upper border, bits at lower border, mean p
LOG_PER_DOUBLING = 8
CHECKPOINT_SECONDS = 60.0
CHUNK_SECONDS = 0.05  # target duration of one kernel call (keeps the current row and stop requests responsive)

# Header layout: 32 int64 values, then 32 float64 values.
H_MAGIC, H_N, H_T, H_STATUS, H_RUNTIME, H_RECENT_COUNT, H_HIST_COUNT, H_HIST_INTERVAL, H_LOG_COUNT, \
    H_HEARTBEAT_NS, H_RECENT_K, H_SEQ, H_CUR_T = range(13)
F_RATE, F_ELAPSED = range(2)


def log_schedule(count: int) -> np.ndarray:
    """Distinct integer times 0, 1, 2, 3, ..., ~2^(j/8): the rows of the log-time history."""
    times, j = [0], 0
    while len(times) < count:
        t = min(int(math.ceil(2 ** (j / LOG_PER_DOUBLING))), 2**62 + len(times))
        if t > times[-1]:
            times.append(t)
        j += 1
    return np.array(times, dtype=np.int64)


@dataclass(frozen=True)
class Layout:
    n: int
    R: int  # cascade rows
    H: int  # compacted-history rows (even)
    G: int  # log-time rows

    @staticmethod
    def for_n(n: int) -> "Layout":
        R = int(min(600, max(100, 32_000_000 // n)))
        H = int(min(2048, max(256, 64_000_000 // n))) // 2 * 2
        return Layout(n, R, H, 400)  # 400 log-time rows reach t ~ 2.8e14

    def sections(self) -> list[tuple[str, np.dtype, tuple]]:
        n, R, H, G = self.n, self.R, self.H, self.G
        return [
            ("ihdr", np.dtype(np.int64), (32,)),
            ("fhdr", np.dtype(np.float64), (32,)),
            ("current", np.dtype(np.uint8), (n,)),
            ("recent", np.dtype(np.uint8), (R, n)),
            ("recent_t", np.dtype(np.int64), (R,)),
            ("recent_m", np.dtype(np.float32), (R, N_METRICS)),
            ("hist", np.dtype(np.uint8), (H, n)),
            ("hist_t", np.dtype(np.int64), (H,)),
            ("hist_m", np.dtype(np.float32), (H, N_METRICS)),
            ("log", np.dtype(np.uint8), (G, n)),
            ("log_t", np.dtype(np.int64), (G,)),
            ("log_m", np.dtype(np.float32), (G, N_METRICS)),
        ]


class LiveStore:
    """The memory-mapped file shared by the runner (writer) and the app (reader)."""

    def __init__(self, path: Path, layout: Layout, mode: str):
        self.layout = layout
        offset, self.arrays = 0, {}
        total = sum(dt.itemsize * int(np.prod(shape)) for _, dt, shape in layout.sections())
        if mode == "w+" and path.exists() and path.stat().st_size == total:
            mode = "r+"  # resuming: keep what is there
        self._mm = np.memmap(path, dtype=np.uint8, mode=mode, shape=(total,))
        for name, dt, shape in layout.sections():
            size = dt.itemsize * int(np.prod(shape))
            self.arrays[name] = self._mm[offset:offset + size].view(dt).reshape(shape)
            offset += size

    def __getattr__(self, name):
        try:
            return self.__dict__["arrays"][name]
        except KeyError:
            raise AttributeError(name) from None

    def flush(self) -> None:
        self._mm.flush()


def quantize(p: np.ndarray) -> np.ndarray:
    return np.rint(p * 255.0).astype(np.uint8)


def metrics(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    at_upper = p >= hi
    front = int(np.argmin(at_upper)) if not at_upper.all() else len(p)
    return np.array([front, at_upper.sum(), (p <= lo).sum(), p.mean()], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Runner (background process)
# --------------------------------------------------------------------------- #


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def run(run_dir: Path) -> None:
    """Runner main loop (started in its own process by start() / resume())."""
    meta = json.loads((run_dir / "meta.json").read_text())
    n, L, fitness = int(meta["n"]), float(meta["L"]), meta["fitness"]
    K = float(meta["K"])
    budget = int(meta["max_iterations"]) if meta.get("max_iterations") else None
    rows_per_second = float(meta.get("rows_per_second", 20))
    lo, hi = borders(n, L)
    layout = Layout(**meta["layout"])
    store = LiveStore(run_dir / "live.bin", layout, "w+")
    ih, fh = store.ihdr, store.fhdr
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    ckpt_path = run_dir / "checkpoint.json"
    rng = np.random.default_rng(int(meta["seed"]))
    kern = kernel.Kernel(n, K, L, fitness, rng.bit_generator.state)
    log_t = log_schedule(layout.G)
    if ckpt_path.exists():  # resume exactly where the last checkpoint left off
        ck = json.loads(ckpt_path.read_text())
        state = {"bit_generator": "PCG64", "state": {"state": int(ck["rng_state"]), "inc": int(ck["rng_inc"])},
                 "has_uint32": 0, "uinteger": 0}
        kern.close()
        kern = kernel.Kernel(n, K, L, fitness, state)
        kern.p[:] = np.load(run_dir / "checkpoint_p.npy")
        t, elapsed = int(ck["t"]), float(ck["elapsed"])
        hist_count, hist_interval = int(ck["hist_count"]), int(ck["hist_interval"])
        log_count, recent_count, next_recent, k = (int(ck["log_count"]), int(ck["recent_count"]),
                                                   int(ck["next_recent"]), int(ck["recent_k"]))
    else:
        t, elapsed = 0, 0.0
        hist_count, hist_interval, log_count, recent_count, next_recent, k = 0, 1, 0, 0, 0, 1000
        ih[:] = 0
        fh[:] = 0
    ih[H_MAGIC], ih[H_N], ih[H_RUNTIME] = MAGIC, n, int(ck["runtime"]) if ckpt_path.exists() else -1
    ih[H_STATUS] = RUNNING

    def checkpoint(status_code: Optional[int] = None) -> None:
        np.save(run_dir / "checkpoint_p.tmp.npy", kern.p)
        os.replace(run_dir / "checkpoint_p.tmp.npy", run_dir / "checkpoint_p.npy")
        st = kern.rng_state()["state"]
        _write_json_atomic(ckpt_path, {
            "t": t, "elapsed": elapsed, "rng_state": str(st["state"]), "rng_inc": str(st["inc"]),
            "hist_count": hist_count, "hist_interval": hist_interval, "log_count": log_count,
            "recent_count": recent_count, "next_recent": next_recent, "recent_k": k,
            "runtime": int(ih[H_RUNTIME]),
            "status": STATUS[int(ih[H_STATUS]) if status_code is None else status_code],
        })
        store.flush()

    def record(t_label: int, row: np.ndarray, m: np.ndarray, hist: bool, log: bool, recent: bool) -> None:
        nonlocal hist_count, hist_interval, log_count, recent_count
        if hist:
            store.hist[hist_count], store.hist_t[hist_count], store.hist_m[hist_count] = row, t_label, m
            hist_count += 1
            if hist_count == layout.H:  # compact: keep every other row, double the spacing
                store.hist[: layout.H // 2] = store.hist[0::2]
                store.hist_t[: layout.H // 2] = store.hist_t[0::2]
                store.hist_m[: layout.H // 2] = store.hist_m[0::2]
                hist_count, hist_interval = layout.H // 2, hist_interval * 2
            ih[H_HIST_COUNT], ih[H_HIST_INTERVAL] = hist_count, hist_interval
        if log and log_count < layout.G:
            store.log[log_count], store.log_t[log_count], store.log_m[log_count] = row, t_label, m
            log_count += 1
            ih[H_LOG_COUNT] = log_count
        if recent:
            slot = recent_count % layout.R
            store.recent[slot], store.recent_t[slot], store.recent_m[slot] = row, t_label, m
            recent_count += 1
            ih[H_RECENT_COUNT] = recent_count

    session_start, last_ckpt = time.time(), time.time()
    chunk, rate = 1000, 0.0
    status = RUNNING
    try:
        while True:
            if stop["flag"] or (run_dir / "stop.request").exists():
                status = STOPPED
                break
            next_hist = hist_count * hist_interval
            next_log = int(log_t[log_count]) if log_count < layout.G else None
            # Rows due at the current t (t = 0 at the start, or exactly at an event after a kernel call).
            due_hist, due_log, due_recent = t == next_hist, next_log == t, t >= next_recent
            if due_hist or due_log or due_recent:
                row, m = quantize(kern.p), metrics(kern.p, lo, hi)
                record(t, row, m, due_hist, due_log, due_recent)
                if due_recent:
                    next_recent = t + k
                continue
            target = min(t + chunk, next_hist, next_recent)
            if next_log is not None:
                target = min(target, next_log)
            if budget is not None:
                target = min(target, budget)
            t0 = time.perf_counter()
            done, found = kern.run(target - t)
            dt = time.perf_counter() - t0
            t += done
            if dt > 0 and done > 0:
                rate = done / dt if rate == 0 else 0.8 * rate + 0.2 * done / dt
                if target == t:  # chunk-limited or event-limited: adapt the chunk to ~CHUNK_SECONDS
                    chunk = int(min(max(chunk * CHUNK_SECONDS / max(dt, 1e-6), 100), 1 << 30)) \
                        if done == chunk else chunk
                k = max(1, int(rate / rows_per_second))
            now = time.time()
            elapsed += now - session_start
            session_start = now
            store.current[:] = quantize(kern.p)
            ih[H_T], ih[H_CUR_T], ih[H_RECENT_K], ih[H_HEARTBEAT_NS] = t, t, k, time.time_ns()
            fh[F_RATE], fh[F_ELAPSED] = rate, elapsed
            ih[H_SEQ] += 1
            if found:  # the optimum was sampled in iteration t, from the distribution p^(t-1) shown now
                ih[H_RUNTIME] = t
                row, m = quantize(kern.p), metrics(kern.p, lo, hi)
                record(t - 1, row, m, True, True, True)
                status = FINISHED
                break
            if budget is not None and t >= budget:
                row, m = quantize(kern.p), metrics(kern.p, lo, hi)
                record(t, row, m, True, True, True)
                status = BUDGET
                break
            if now - last_ckpt >= CHECKPOINT_SECONDS:
                checkpoint()
                last_ckpt = now
    except Exception:
        status = ERROR
        raise
    finally:
        # Order matters: the final status is written last, once the checkpoint is complete, so a new
        # runner (resume) can never start while this one is still writing.
        ih[H_T], ih[H_HEARTBEAT_NS] = t, time.time_ns()
        checkpoint(status)
        (run_dir / "stop.request").unlink(missing_ok=True)
        kern.close()
        ih[H_STATUS] = status
        store.flush()


# --------------------------------------------------------------------------- #
# Control (used by the app)
# --------------------------------------------------------------------------- #


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip())[:40].strip("-")


def _launch(run_dir: Path) -> int:
    log = open(run_dir / "runner.log", "a")
    proc = subprocess.Popen([sys.executable, "-m", "cga.live", "run", str(run_dir)], cwd=str(PROJECT),
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    meta = json.loads((run_dir / "meta.json").read_text())
    meta["pid"] = proc.pid
    _write_json_atomic(run_dir / "meta.json", meta)
    return proc.pid


def start(n: int, K_expr: str, L: float, fitness: str, seed: int, max_iterations: Optional[int],
          rows_per_second: float = 20.0, label: str = "") -> Path:
    """Create and launch a new live run; returns its directory."""
    K = eval_expr(K_expr, n)
    validate_params(n, K, L, max_iterations or 1)
    if fitness not in kernel.FITNESS_CODES:
        raise ValueError(f"live runs support {sorted(kernel.FITNESS_CODES)}, not '{fitness}'")
    kernel.load()  # fail here (with a clear message) rather than in the background process
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{_slug(label)}" if _slug(label) else stamp
    run_dir = LIVE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    layout = Layout.for_n(n)
    meta = {"id": run_id, "label": label.strip(), "n": n, "K_expr": str(K_expr), "K": K, "L": L,
            "fitness": fitness, "seed": seed, "max_iterations": max_iterations or None,
            "rows_per_second": rows_per_second, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "layout": layout.__dict__}
    _write_json_atomic(run_dir / "meta.json", meta)
    _launch(run_dir)
    return run_dir


def is_alive(run_dir: Path) -> bool:
    """True while the run's runner process is working on it."""
    try:
        pid = int(json.loads((run_dir / "meta.json").read_text()).get("pid", 0))
        if pid <= 0:
            return False
        try:  # a finished runner started by this process stays a zombie until reaped
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return False
        except ChildProcessError:
            pass  # not our child (e.g. started by an earlier app session)
        os.kill(pid, 0)
    except (OSError, ValueError, FileNotFoundError):
        return False
    try:
        ih = open_store(run_dir).ihdr
        if int(ih[H_STATUS]) not in (STARTING, RUNNING):
            return False  # the runner has finished (it writes its final status last)
        # The pid could have been reused by another process: also require a recent heartbeat
        # (none yet means the runner is still starting).
        hb = int(ih[H_HEARTBEAT_NS])
    except (OSError, ValueError, FileNotFoundError):
        return True
    return hb == 0 or time.time_ns() - hb < 30e9


def stop(run_dir: Path) -> None:
    (run_dir / "stop.request").touch()


def resume(run_dir: Path) -> None:
    if is_alive(run_dir):
        return
    (run_dir / "stop.request").unlink(missing_ok=True)
    _launch(run_dir)


def delete(run_dir: Path) -> None:
    if is_alive(run_dir):
        stop(run_dir)
        for _ in range(50):
            if not is_alive(run_dir):
                break
            time.sleep(0.1)
    shutil.rmtree(run_dir, ignore_errors=True)


def list_runs() -> list[Path]:
    if not LIVE_ROOT.exists():
        return []
    return sorted((d for d in LIVE_ROOT.iterdir() if (d / "meta.json").exists()), reverse=True)


def read_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text())


def open_store(run_dir: Path) -> LiveStore:
    meta = read_meta(run_dir)
    return LiveStore(run_dir / "live.bin", Layout(**meta["layout"]), "r")


def status_of(run_dir: Path) -> str:
    """Status for display; a run whose process died without stopping cleanly is 'interrupted'."""
    try:
        code = int(open_store(run_dir).ihdr[H_STATUS])
    except (FileNotFoundError, ValueError):
        return "starting" if is_alive(run_dir) else "interrupted"
    if code in (STARTING, RUNNING) and not is_alive(run_dir):
        return "interrupted"
    return STATUS.get(code, "unknown")


def snapshot(run_dir: Path) -> dict:
    """A consistent-enough copy of everything the viewer needs (arrays copied out of the mmap)."""
    st = open_store(run_dir)
    ih, fh = st.ihdr.copy(), st.fhdr.copy()
    lay = st.layout
    rc = int(ih[H_RECENT_COUNT])
    m = min(rc, lay.R)
    order = [(rc - 1 - i) % lay.R for i in range(m)]  # newest first
    hc, gc = int(ih[H_HIST_COUNT]), int(ih[H_LOG_COUNT])
    return {
        "t": int(ih[H_T]), "runtime": int(ih[H_RUNTIME]), "rate": float(fh[F_RATE]),
        "elapsed": float(fh[F_ELAPSED]), "k": int(ih[H_RECENT_K]), "hist_interval": int(ih[H_HIST_INTERVAL]),
        "current": st.current.copy(),
        "recent": st.recent[order].copy(), "recent_t": st.recent_t[order].copy(), "recent_m": st.recent_m[order].copy(),
        "hist": st.hist[:hc].copy(), "hist_t": st.hist_t[:hc].copy(), "hist_m": st.hist_m[:hc].copy(),
        "log": st.log[:gc].copy(), "log_t": st.log_t[:gc].copy(), "log_m": st.log_m[:gc].copy(),
    }


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        run(Path(sys.argv[2]))
    else:
        print("usage: python -m cga.live run <run dir>", file=sys.stderr)
        sys.exit(2)
