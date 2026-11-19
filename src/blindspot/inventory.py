"""Step 10: inventory simulation (single-day newsvendor for perishable products).

Stated assumptions (the dataset has no prices, costs or lead times)
- Fresh products, ordered daily, delivered next morning (lead time = 1 day = the forecast horizon).
- Unsold stock is wasted at the end of the day (no carry-over); unmet demand is lost.
- Costs are relative: a lost sale costs `cu` per unit (lost margin), a wasted unit costs `co` (purchase cost).
  Default cu = 1.0, co = 0.5 -> critical ratio CR = cu / (cu + co) = 2/3.
- Order rule (same for every model): Q = max(0, forecast + safety), where `safety` is the CR-quantile of the
  model's own forecast errors on its validation days, separately for low- and high-demand series.
  Each model's errors are measured against ITS OWN training target, exactly as a planner using that data would
  (so B's safety allowance is calibrated on censored sales).
- Outcomes are scored against TRUE demand on clean test days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CU, CO = 1.0, 0.5
CR_GRID = [0.5, 2 / 3, 0.8, 0.9]


def critical_ratio(cu: float = CU, co: float = CO) -> float:
    return cu / (cu + co)


def costs_for_cr(cr: float, cu: float = CU) -> tuple[float, float]:
    """(cu, co) giving critical ratio `cr` with the shortage cost fixed at `cu`."""
    return cu, cu * (1 - cr) / cr


def safety_quantiles(resid: np.ndarray, groups: np.ndarray, cr: float) -> dict:
    """CR-quantile of forecast errors (actual - forecast) per group."""
    s = pd.Series(np.asarray(resid, float))
    return {g: float(s[groups == g].quantile(cr)) for g in pd.unique(groups)}


def order_quantity(forecast: np.ndarray, groups: np.ndarray, safety: dict) -> np.ndarray:
    add = pd.Series(groups).map(safety).to_numpy(float)
    return np.clip(np.asarray(forecast, float) + add, 0, None)


def outcomes(Q: np.ndarray, D: np.ndarray, cu: float = CU, co: float = CO) -> pd.DataFrame:
    """Per-row newsvendor outcome of ordering Q when demand is D."""
    Q, D = np.asarray(Q, float), np.asarray(D, float)
    short = np.clip(D - Q, 0, None)
    waste = np.clip(Q - D, 0, None)
    return pd.DataFrame({"order": Q, "demand": D, "sold": np.minimum(Q, D), "short": short, "waste": waste,
                         "stockout": (short > 1e-9).astype(int), "cost": cu * short + co * waste})


def summarize(o: pd.DataFrame) -> dict:
    d = o["demand"].sum()
    return {"fill_rate": o["sold"].sum() / d, "stockout_rate": o["stockout"].mean(),
            "short_per_demand": o["short"].sum() / d, "waste_per_demand": o["waste"].sum() / d,
            "cost_per_demand": o["cost"].sum() / d, "order_per_demand": o["order"].sum() / d}
