"""Forecast metrics, following the frn-50k-baseline evaluation protocol.

Protocol (frn-50k-baseline, Apache 2.0): score only stockout-free test days; compute each metric
per test date over all series, then average over the dates. Groups: all, low-sales (mean daily
train sales < 1) and high-sales (>= 1).

  WAPE = sum|pred - y| / sum y         main error metric
  WPE  = sum(pred - y) / sum y         bias (negative = under-forecast)
  MAE  = mean|pred - y|
  RMSE = sqrt(mean (pred - y)^2)       added by BlindSpot
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daily(df: pd.DataFrame, pred: str, y: str) -> pd.DataFrame:
    e = df[pred] - df[y]
    g = pd.DataFrame({"dt": df["dt"], "abs": e.abs(), "err": e, "sq": e ** 2, "y": df[y]}).groupby("dt")
    s = g.sum()
    n = g.size()
    return pd.DataFrame({"wape": s["abs"] / s["y"], "wpe": s["err"] / s["y"], "mae": s["abs"] / n,
                         "rmse": np.sqrt(s["sq"] / n)})


def evaluate(df: pd.DataFrame, pred: str, y: str = "sales_true") -> pd.DataFrame:
    """df: test rows with columns dt, sales_group, is_clean_day, <pred>, <y>. Returns one row per group."""
    d = df[df["is_clean_day"] == 1]
    rows = []
    for name, sub in [("all", d), ("low", d[d.sales_group == "low"]), ("high", d[d.sales_group == "high"])]:
        m = _daily(sub, pred, y).mean()
        rows.append({"group": name, "n_days": len(sub), **m.round(4).to_dict()})
    return pd.DataFrame(rows)


def bootstrap_wape(df: pd.DataFrame, pred: str, y: str = "sales_true", n_boot: int = 500, seed: int = 0,
                   cluster: str = "store_id") -> tuple[float, float]:
    """95% CI of overall WAPE (pooled), resampling stores with replacement (series within a store are correlated)."""
    d = df[df["is_clean_day"] == 1]
    agg = d.assign(a=(d[pred] - d[y]).abs()).groupby(cluster)[["a", y]].sum()
    a, t = agg["a"].to_numpy(), agg[y].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(agg), size=(n_boot, len(agg)))
    w = a[idx].sum(axis=1) / t[idx].sum(axis=1)
    return float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))
