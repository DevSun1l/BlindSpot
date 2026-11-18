"""Step 3: build the three training-data conditions.

Only clean (stockout-free) TRAIN days are candidates for masking; their recorded sales are
treated as true demand. Real stockout days and all eval (test) days are identical in every
condition, so the conditions differ only on the masked days.

  A  clean           masked days keep their true sales
  B  censored        masked days keep only the sales made before a simulated stockout hour
                     (stockout lasts until the end of the day, as in 87% of real stockouts)
  C  matched noise   masked days get the SAME absolute distortion as in B, with a random sign

Masking is demand-linked (design option B): within each series, a day's chance of being masked
rises with its sales percentile, mimicking real stockouts, which happen more on high-demand days.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from blindspot.data import BUSINESS_HOURS


@dataclass(frozen=True)
class ConditionConfig:
    mask_rate: float = 0.30      # share of clean train days that get masked
    demand_power: float = 1.0    # 0 = random masking; higher = stronger link to high-demand days
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


def stockout_start_distribution(panel: pd.DataFrame) -> pd.Series:
    """Empirical distribution of the first stockout hour (6..21) on real stockout TRAIN days."""
    d = panel[(panel.split == "train") & (panel.is_stockout_day == 1)]
    st = np.stack(d.hours_stock_status.to_numpy())[:, BUSINESS_HOURS]
    first = st.argmax(axis=1) + BUSINESS_HOURS.start
    return pd.Series(first).value_counts(normalize=True).sort_index()


def mask_probabilities(sales: pd.Series, series_id: pd.Series, rate: float, power: float) -> np.ndarray:
    """Per-day masking probability, proportional to (within-series sales percentile) ** power,
    rescaled per series so its expected masked share equals `rate`. Clipped to [0, 1]."""
    pct = sales.groupby(series_id).rank(pct=True, method="average").to_numpy()
    w = pct ** power
    mean_w = pd.Series(w).groupby(series_id.to_numpy()).transform("mean").to_numpy()
    return np.clip(rate * w / mean_w, 0.0, 1.0)


def censor_day(hours_sale: np.ndarray, start_hour: int) -> float:
    """Sales observed if stock runs out at `start_hour` and stays out for the rest of the day."""
    return float(np.sum(hours_sale[:start_hour]))


def build_conditions(panel: pd.DataFrame, cfg: ConditionConfig = ConditionConfig()) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(cfg.seed)
    df = panel[["series_id", "store_id", "product_id", "dt", "day_idx", "split", "sale_amount",
                "is_clean_day", "is_stockout_day", "stockout_hours", "hours_sale"]].copy()
    df = df.sort_values(["series_id", "dt"]).reset_index(drop=True)

    candidate = ((df.split == "train") & (df.is_clean_day == 1)).to_numpy()
    prob = np.zeros(len(df))
    cand = df[candidate]
    prob[candidate] = mask_probabilities(cand.sale_amount, cand.series_id, cfg.mask_rate, cfg.demand_power)
    masked = rng.random(len(df)) < prob

    start_dist = stockout_start_distribution(panel)
    start = np.full(len(df), -1, dtype=np.int16)
    start[masked] = rng.choice(start_dist.index.to_numpy(), size=masked.sum(), p=start_dist.to_numpy())

    true = df.sale_amount.to_numpy()
    hs = df.hours_sale.to_numpy()
    censored = true.copy()
    idx = np.flatnonzero(masked)
    censored[idx] = [censor_day(hs[i], start[i]) for i in idx]

    distortion = np.abs(true - censored)                 # >= 0, nonzero only on masked days
    sign = rng.choice([-1.0, 1.0], size=len(df))
    noised = true + sign * distortion                    # never < 0 because distortion <= true

    df["is_masked"] = masked.astype("int8")
    df["mask_prob"] = prob.astype("float32")
    df["sim_stockout_hour"] = start
    df["sales_true"] = true
    df["sales_A"] = true
    df["sales_B"] = censored
    df["sales_C"] = noised
    return df.drop(columns=["hours_sale"]), start_dist
