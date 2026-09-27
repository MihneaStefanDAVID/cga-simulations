"""Evaluate parameter expressions such as "5 * log(n)" from the YAML config."""

from __future__ import annotations

import math
from typing import Union

# log is the natural logarithm, as in math.log.
_NAMESPACE = {
    "log": math.log,
    "ln": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
    "pi": math.pi,
    "e": math.e,
}


def eval_expr(expr: Union[str, int, float], n: int) -> float:
    """Evaluate a number or an expression in n (e.g. "sqrt(n*log(n))") to a float."""
    if isinstance(expr, bool):
        raise ValueError(f"invalid parameter value {expr!r}")
    if isinstance(expr, (int, float)):
        return float(expr)
    namespace = dict(_NAMESPACE, n=n)
    try:
        value = eval(str(expr), {"__builtins__": {}}, namespace)  # trusted local config
    except Exception as exc:
        raise ValueError(f"cannot evaluate expression {expr!r} with n={n}: {exc}") from exc
    return float(value)
