"""Single-period newsvendor inventory simulation for perishable products.

Model assumptions (the dataset does not include prices, costs, or lead times):
- Daily ordering with a 1-day lead time matching the forecast horizon.
- Unsold stock spoils at end of day; unmet demand is lost.
- Relative unit costs: underage (lost sale) `cu` and overage (waste) `co`,
  with default critical ratio `CR = cu / (cu + co) = 2/3`.
- Order quantity `Q = max(0, forecast + safety)`, where `safety` is the
  CR-quantile of each model's validation residuals by sales group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CU, CO = 1.0, 0.5
CR_GRID = [0.5, 2 / 3, 0.8, 0.9]


def critical_ratio(cu: float = CU, co: float = CO) -> float:
    """Newsvendor critical fractile `cu / (cu + co)`."""
    return cu / (cu + co)


def costs_for_cr(cr: float, cu: float = CU) -> tuple[float, float]:
    """Return `(cu, co)` yielding critical ratio `cr` for a fixed shortage cost `cu`."""
    return cu, cu * (1 - cr) / cr


def safety_quantiles(resid: np.ndarray, groups: np.ndarray, cr: float) -> dict:
    """Compute the `cr`-quantile of validation residuals (`actual - forecast`) per group."""
    s = pd.Series(np.asarray(resid, float))
    return {g: float(s[groups == g].quantile(cr)) for g in pd.unique(groups)}


def order_quantity(forecast: np.ndarray, groups: np.ndarray, safety: dict) -> np.ndarray:
    """Compute non-negative order quantities by adding group-specific safety allowances to forecasts."""
    missing = set(pd.unique(groups)) - set(safety)
    if missing:
        raise ValueError(f"no safety allowance for sales group(s) {sorted(missing)}; got {sorted(safety)}")
    add = pd.Series(groups).map(safety).to_numpy(float)
    return np.clip(np.asarray(forecast, float) + add, 0, None)


def outcomes(Q: np.ndarray, D: np.ndarray, cu: float = CU, co: float = CO) -> pd.DataFrame:
    """Evaluate per-row sales, shortages, waste, stockout flags, and total cost for orders `Q` and demand `D`."""
    Q, D = np.asarray(Q, float), np.asarray(D, float)
    short = np.clip(D - Q, 0, None)
    waste = np.clip(Q - D, 0, None)
    return pd.DataFrame({"order": Q, "demand": D, "sold": np.minimum(Q, D), "short": short, "waste": waste,
                         "stockout": (short > 1e-9).astype(int), "cost": cu * short + co * waste})


def summarize(o: pd.DataFrame) -> dict:
    """Aggregate newsvendor outcomes into fill rate, stockout rate, and per-demand cost/shortage/waste ratios."""
    d = o["demand"].sum()
    return {"fill_rate": o["sold"].sum() / d, "stockout_rate": o["stockout"].mean(),
            "short_per_demand": o["short"].sum() / d, "waste_per_demand": o["waste"].sum() / d,
            "cost_per_demand": o["cost"].sum() / d, "order_per_demand": o["order"].sum() / d}
