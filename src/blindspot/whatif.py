"""What-if scenario evaluation on controllable promotion levers.

Restricts interventions to planned promotion inputs (`discount`, `activity_flag`,
and `test_promo`) within their observed training support so scenarios cannot alter
historical sales lags or extrapolate into unsupported discount regions.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from blindspot.faithfulness import PROMO_FEATURE
from blindspot.features import FEATURES
from blindspot.model import predict

# Supported intervention bounds (discounts below 0.5 represent <0.5% of training rows).
LEVER_LIMITS: dict[str, tuple[float, float]] = {"discount": (0.5, 1.0), "activity_flag": (0, 1), PROMO_FEATURE: (0, 1)}
BINARY_LEVERS = {"activity_flag", PROMO_FEATURE}

# Named scenarios as (baseline_overrides, scenario_overrides) dicts; omitted keys keep row values.
SCENARIOS: dict[str, tuple[dict, dict]] = {
    "activity on (discount kept)": ({"activity_flag": 0}, {"activity_flag": 1}),
    "20% discount (no activity)": ({"activity_flag": 0, "discount": 1.0}, {"activity_flag": 0, "discount": 0.8}),
    "promo: activity + 20% discount": ({"activity_flag": 0, "discount": 1.0}, {"activity_flag": 1, "discount": 0.8}),
}
DISCOUNT_GRID = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]


def check_changes(changes: dict, features: list[str]) -> None:
    """Validate that `changes` only modifies allowed levers within their valid range."""
    for f, v in changes.items():
        if f not in LEVER_LIMITS:
            raise ValueError(f"'{f}' cannot be changed in a what-if: only {sorted(LEVER_LIMITS)} are planned levers "
                             "(sales history and other inputs are fixed).")
        if f not in features:
            raise ValueError(f"'{f}' is not a feature of this model.")
        lo, hi = LEVER_LIMITS[f]
        vals = np.atleast_1d(np.asarray(v, dtype=float))
        if np.any(vals < lo) or np.any(vals > hi):
            raise ValueError(f"{f} must be between {lo} and {hi} (training support); got {v}.")
        if f in BINARY_LEVERS and not np.all(np.isin(vals, [0, 1])):
            raise ValueError(f"{f} must be 0 or 1; got {v}.")


def apply_changes(X: pd.DataFrame, changes: dict, features: list[str] | None = None) -> pd.DataFrame:
    """Return a copy of `X[features]` with validated lever overrides applied."""
    feats = features or FEATURES
    check_changes(changes, feats)
    Z = X[feats].copy()
    for f, v in changes.items():
        Z[f] = v
    return Z


def whatif(booster: lgb.Booster, X: pd.DataFrame, baseline: dict, scenario: dict,
           features: list[str] | None = None) -> pd.DataFrame:
    """Compute per-row baseline forecast, scenario forecast, and uplift (`scenario - base`)."""
    feats = features or FEATURES
    f0 = predict(booster, apply_changes(X, baseline, feats), feats)
    f1 = predict(booster, apply_changes(X, scenario, feats), feats)
    return pd.DataFrame({"base": f0, "scenario": f1, "uplift": f1 - f0}, index=X.index)


def discount_curve(booster: lgb.Booster, X: pd.DataFrame, grid: list[float] = DISCOUNT_GRID,
                   features: list[str] | None = None) -> pd.Series:
    """Compute mean forecast across `X` at each discount level in `grid`, holding other features fixed."""
    feats = features or FEATURES
    return pd.Series({d: predict(booster, apply_changes(X, {"discount": d}, feats), feats).mean() for d in grid})


def bootstrap_ratio_ci(num: np.ndarray, den: np.ndarray, clusters: np.ndarray, n_boot: int = 500,
                       seed: int = 0) -> tuple[float, float]:
    """Return a 95% cluster-bootstrap CI for `sum(num) / sum(den)`, resampling `clusters` with replacement."""
    g = pd.DataFrame({"n": num, "d": den, "c": clusters}).groupby("c")[["n", "d"]].sum()
    n, d = g["n"].to_numpy(), g["d"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    r = n[idx].sum(axis=1) / d[idx].sum(axis=1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def observed_activity_ratio(df: pd.DataFrame, y: str = "target") -> float:
    """Median within-series ratio of mean sales on activity days to non-activity days (descriptive only)."""
    g = df.groupby(["series_id", "activity_flag"], observed=True)[y].mean().unstack()
    g = g.dropna()
    g = g[g[0] > 0]
    return float((g[1] / g[0]).median())
