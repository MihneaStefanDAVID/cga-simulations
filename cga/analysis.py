"""Statistics on sweep/scaling summaries: empirical power-law exponents T ~ n^b."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def fit_power_law(n: np.ndarray, y: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Least-squares fit of log(y) = a + b*log(n) on the finite, positive entries.

    Returns (b, a, r_squared), or None if fewer than 2 usable points. b is the
    empirical polynomial exponent: y ~ e^a * n^b. With exactly 2 points the fit is
    always perfect (R^2 = 1), so b says little unless several n values are used.
    """
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & (y > 0) & (n > 0)
    if mask.sum() < 2:
        return None
    x = np.log(n[mask])
    ly = np.log(y[mask])
    b, a = np.polyfit(x, ly, 1)
    pred = a + b * x
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(b), float(a), float(r2)


def scaling_fits(summary: pd.DataFrame, K_exprs: Sequence[str]) -> pd.DataFrame:
    """Fit T ~ n^b per K expression, using only budget-unbiased medians.

    The statistic is `median_censored` (median over *all* repetitions, unfinished
    runs counted as +inf). It is defined exactly where more than half of the runs
    finished, and there it is not biased by the budget. Points where at most half
    finished are left out: the median over the finished runs alone would be biased
    low there, and so would the exponent.
    """
    rows = []
    for k_expr in K_exprs:
        s = summary[summary["K_expr"] == k_expr].sort_values("n")
        n = s["n"].to_numpy(float)
        med = s["median_censored"].to_numpy(float)
        used = np.isfinite(med) & (med > 0)
        fit = fit_power_law(n, med)
        row = {
            "K_expr": k_expr,
            "exponent_b": np.nan,
            "prefactor": np.nan,
            "r_squared": np.nan,
            "n_points_used": int(used.sum()),
            "n_points_total": len(n),
            "n_min_used": n[used].min() if used.any() else np.nan,
            "n_max_used": n[used].max() if used.any() else np.nan,
        }
        if fit is not None:
            b, a, r2 = fit
            row.update(exponent_b=b, prefactor=float(np.exp(a)), r_squared=r2)
        rows.append(row)
    return pd.DataFrame(rows)
