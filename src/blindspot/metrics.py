"""Forecast evaluation metrics following the frn-50k-baseline protocol.

Scores stockout-free test days only, computing each metric per date across all
series and then averaging across dates for all, low-sales, and high-sales groups:
  WAPE = sum|pred - y| / sum y
  WPE  = sum(pred - y) / sum y   (negative indicates under-forecasting)
  MAE  = mean|pred - y|
  RMSE = sqrt(mean((pred - y)^2))
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daily(df: pd.DataFrame, pred: str, y: str) -> pd.DataFrame:
    """Compute per-date error metrics across all rows on each date."""
    e = df[pred] - df[y]
    g = pd.DataFrame({"dt": df["dt"], "abs": e.abs(), "err": e, "sq": e ** 2, "y": df[y]}).groupby("dt")
    s = g.sum()
    n = g.size()
    return pd.DataFrame({"wape": s["abs"] / s["y"], "wpe": s["err"] / s["y"], "mae": s["abs"] / n,
                         "rmse": np.sqrt(s["sq"] / n)})


def evaluate(df: pd.DataFrame, pred: str, y: str = "sales_true") -> pd.DataFrame:
    """Evaluate `pred` against `y` on clean days (`is_clean_day == 1`), returning one row per sales group."""
    d = df[df["is_clean_day"] == 1]
    rows = []
    for name, sub in [("all", d), ("low", d[d.sales_group == "low"]), ("high", d[d.sales_group == "high"])]:
        m = _daily(sub, pred, y).mean()
        rows.append({"group": name, "n_days": len(sub), **m.round(4).to_dict()})
    return pd.DataFrame(rows)


def bootstrap_wape(df: pd.DataFrame, pred: str, y: str = "sales_true", n_boot: int = 500, seed: int = 0,
                   cluster: str = "store_id") -> tuple[float, float]:
    """Return a 95% cluster-bootstrap CI for WAPE as reported by `evaluate` (mean over dates of each date's WAPE),
    resampling `cluster` (stores) with replacement."""
    d = df[df["is_clean_day"] == 1]
    # Per cluster and date: sums of |error| and of y, so within-store correlation is preserved.
    agg = d.assign(a=(d[pred] - d[y]).abs()).groupby([cluster, "dt"], observed=True)[["a", y]].sum()
    a = agg["a"].unstack("dt", fill_value=0.0).to_numpy()          # (n_clusters, n_dates)
    t = agg[y].unstack("dt", fill_value=0.0).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.shape[0], size=(n_boot, a.shape[0]))
    w = (a[idx].sum(axis=1) / t[idx].sum(axis=1)).mean(axis=1)    # daily WAPE per draw, then mean over dates
    return float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))
