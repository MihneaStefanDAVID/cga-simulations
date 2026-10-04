"""C++ cGA kernel: compiled on first use, loaded with ctypes, bit-for-bit identical to Python.

The kernel (cga_kernel.cpp) reproduces numpy's PCG64 stream and the exact arithmetic of
cga/simulator.py, so for the same seed it returns exactly the same runtime and frequency
vectors as run_cga, only faster. tests/test_simulator.py checks this equality.

Engine selection (environment variable CGA_ENGINE): "auto" (default: C++ when it can be built,
otherwise Python), "cpp" (require C++), "python" (always use the Python implementation).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..simulator import RunResult, borders, validate_params

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "cga_kernel.cpp"
BUILD_DIR = HERE / "build"
FITNESS_CODES = {"binval": 0, "onemax": 1}  # fitness functions implemented in C++
MASK64 = (1 << 64) - 1

_lib: Optional[ctypes.CDLL] = None
_load_error: Optional[str] = None


def _library_path() -> Path:
    digest = hashlib.sha1(SOURCE.read_bytes()).hexdigest()[:12]  # rebuild when the source changes
    ext = "dylib" if sys.platform == "darwin" else "so"
    return BUILD_DIR / f"libcga_kernel-{digest}-{platform.machine()}.{ext}"


def _compile(target: Path) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cxx = os.environ.get("CXX", "c++")
    arch = ["-mcpu=native"] if platform.machine() in ("arm64", "aarch64") else ["-march=native"]
    base = [cxx, "-O3", "-std=c++17", "-shared", "-fPIC"]
    # Compile to a temporary file and rename, so concurrent worker processes never load a partial file.
    fd, tmp = tempfile.mkstemp(dir=BUILD_DIR, suffix=target.suffix)
    os.close(fd)
    try:
        for flags in (arch, []):  # retry without CPU-specific flags if the compiler rejects them
            res = subprocess.run(base + flags + ["-o", tmp, str(SOURCE)], capture_output=True, text=True)
            if res.returncode == 0:
                os.replace(tmp, target)
                return
        raise RuntimeError(f"compiling the C++ kernel failed:\n{res.stderr.strip()}")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load() -> ctypes.CDLL:
    """Compile (if needed) and load the kernel library. Raises RuntimeError if impossible."""
    global _lib, _load_error
    if _lib is not None:
        return _lib
    if _load_error is not None:
        raise RuntimeError(_load_error)
    try:
        path = _library_path()
        if not path.exists():
            _compile(path)
        lib = ctypes.CDLL(str(path))
    except Exception as exc:  # no compiler, compile error, unloadable library
        _load_error = f"C++ kernel unavailable: {exc}"
        raise RuntimeError(_load_error) from exc
    u64, i64, i32 = ctypes.c_uint64, ctypes.c_int64, ctypes.c_int32
    lib.cga_create.restype = ctypes.c_void_p
    lib.cga_create.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                               u64, u64, u64, u64]
    lib.cga_destroy.argtypes = [ctypes.c_void_p]
    lib.cga_p.restype = ctypes.POINTER(ctypes.c_double)
    lib.cga_p.argtypes = [ctypes.c_void_p]
    lib.cga_run.restype = i64
    lib.cga_run.argtypes = [ctypes.c_void_p, i64, ctypes.POINTER(i32)]
    lib.cga_rng_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(u64)]
    _lib = lib
    return lib


def available() -> bool:
    try:
        load()
        return True
    except RuntimeError:
        return False


def engine_for(fitness: str) -> str:
    """'cpp' or 'python', according to CGA_ENGINE and what the C++ kernel supports."""
    choice = os.environ.get("CGA_ENGINE", "auto").lower()
    if choice not in ("auto", "cpp", "python"):
        raise ValueError(f"CGA_ENGINE must be auto, cpp or python, got {choice!r}")
    if choice == "python" or fitness not in FITNESS_CODES:
        if choice == "cpp" and fitness not in FITNESS_CODES:
            raise RuntimeError(f"the C++ kernel does not implement fitness '{fitness}'")
        return "python"
    if choice == "cpp":
        load()  # raise if unavailable
        return "cpp"
    return "cpp" if available() else "python"


class Kernel:
    """One cGA state (frequencies + random stream) living in C++.

    `p` is a numpy view of the kernel's frequency vector (read it, or write it before running).
    """

    def __init__(self, n: int, K: float, L: float, fitness: str, rng_state: dict):
        lib = load()
        if fitness not in FITNESS_CODES:
            raise ValueError(f"the C++ kernel does not implement fitness '{fitness}'")
        if rng_state.get("bit_generator") != "PCG64":
            raise ValueError("the C++ kernel needs a numpy PCG64 generator (np.random.default_rng)")
        lo, hi = borders(n, L)
        s, inc = rng_state["state"]["state"], rng_state["state"]["inc"]
        self._lib = lib
        self._h = lib.cga_create(n, 1.0 / K, lo, hi, FITNESS_CODES[fitness],
                                 s >> 64, s & MASK64, inc >> 64, inc & MASK64)
        if not self._h:
            raise ValueError("invalid kernel parameters")
        self.n = n
        self.p = np.ctypeslib.as_array(lib.cga_p(self._h), shape=(n,))
        self._found = ctypes.c_int32(0)

    def run(self, max_iters: int) -> tuple[int, bool]:
        """Run up to max_iters iterations; returns (iterations executed, optimum sampled)."""
        done = self._lib.cga_run(self._h, int(max_iters), ctypes.byref(self._found))
        return int(done), bool(self._found.value)

    def rng_state(self) -> dict:
        """numpy PCG64 state after the random numbers consumed so far."""
        out = (ctypes.c_uint64 * 4)()
        self._lib.cga_rng_state(self._h, out)
        return {"bit_generator": "PCG64",
                "state": {"state": (out[0] << 64) | out[1], "inc": (out[2] << 64) | out[3]},
                "has_uint32": 0, "uinteger": 0}

    def close(self) -> None:
        if getattr(self, "_h", None):
            self.p = None
            self._lib.cga_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()


def run_cga_cpp(
    n: int,
    K: float,
    L: float,
    fitness: str,
    max_iterations: int,
    rng: np.random.Generator,
    log_checkpoints: bool = False,
    on_progress: Optional[Callable[[int], None]] = None,
    progress_every: int = 1 << 16,
) -> RunResult:
    """Same contract and same results as simulator.run_cga, computed by the C++ kernel.

    The generator `rng` is advanced exactly as run_cga would advance it.
    """
    validate_params(n, K, L, max_iterations)
    kern = Kernel(n, K, L, fitness, rng.bit_generator.state)
    try:
        times: list[int] = []
        snaps: list[np.ndarray] = []
        if log_checkpoints:
            times.append(0)
            snaps.append(kern.p.copy())
        next_ckpt = 1
        next_report = progress_every if on_progress is not None else max_iterations + 1
        t = 0
        converged = False
        while t < max_iterations:
            target = min(max_iterations, next_report, next_ckpt if log_checkpoints else max_iterations)
            done, found = kern.run(target - t)
            t += done
            if found:
                converged = True
                if log_checkpoints and times[-1] != t - 1:  # the distribution the optimum came from
                    times.append(t - 1)
                    snaps.append(kern.p.copy())
                break
            if t == next_report:
                on_progress(t)
                next_report += progress_every
            if log_checkpoints and t == next_ckpt:
                times.append(t)
                snaps.append(kern.p.copy())
                next_ckpt *= 2
        if not converged and log_checkpoints and times[-1] != max_iterations:
            times.append(max_iterations)
            snaps.append(kern.p.copy())
        rng.bit_generator.state = kern.rng_state()
    finally:
        kern.close()
    if not log_checkpoints:
        return RunResult(converged, t if converged else None, t)
    return RunResult(converged, t if converged else None, t, np.asarray(times), np.vstack(snaps))
