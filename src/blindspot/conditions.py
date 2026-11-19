"""Construct clean (A), stockout-censored (B), and matched-noise (C) training conditions.

Only stockout-free training days are eligible for masking; their recorded sales
serve as ground-truth demand. Real stockout days and all evaluation days remain
identical across conditions:

  A  clean          masked days retain true sales
  B  censored       masked days retain only sales before a sampled stockout hour
                    (assuming the stockout persists through end of day)
  C  matched noise  masked days receive the same distortion magnitude as B with random sign

Masking probabilities increase with each day's within-series sales percentile to
reflect higher stockout risk on high-demand days.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from blindspot.data import BUSINESS_HOURS


@dataclass(frozen=True)
class ConditionConfig:
    mask_rate: float = 0.30    # Target fraction of clean training days to mask
    demand_power: float = 1.0  # Exponent on sales percentile (0 = uniform random masking)
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


def stockout_start_distribution(panel: pd.DataFrame) -> pd.Series:
    """Empirical distribution of the first stockout hour (6..21) on real stockout training days."""
    d = panel[(panel.split == "train") & (panel.is_stockout_day == 1)]
    st = np.stack(d.hours_stock_status.to_numpy())[:, BUSINESS_HOURS]
    # argmax returns the first 1 (out-of-stock hour) within the 6..21 slice.
    first = st.argmax(axis=1) + BUSINESS_HOURS.start
    return pd.Series(first).value_counts(normalize=True).sort_index()


def mask_probabilities(sales: pd.Series, series_id: pd.Series, rate: float, power: float) -> np.ndarray:
    """Compute per-day masking probabilities proportional to (within-series sales percentile) ** power,
    rescaled per series so expected masked share matches `rate` (clipped to [0, 1])."""
    pct = sales.groupby(series_id).rank(pct=True, method="average").to_numpy()
    w = pct ** power
    mean_w = pd.Series(w).groupby(series_id.to_numpy()).transform("mean").to_numpy()
    return np.clip(rate * w / mean_w, 0.0, 1.0)


def censor_day(hours_sale: np.ndarray, start_hour: int) -> float:
    """Total sales observed before `start_hour`, assuming zero sales from `start_hour` onward."""
    return float(np.sum(hours_sale[:start_hour]))


def build_conditions(panel: pd.DataFrame, cfg: ConditionConfig = ConditionConfig()) -> tuple[pd.DataFrame, pd.Series]:
    """Generate condition sales columns (sales_A, sales_B, sales_C) and the empirical start-hour distribution."""
    rng = np.random.default_rng(cfg.seed)
    df = panel[["series_id", "store_id", "product_id", "dt", "day_idx", "split", "sale_amount",
                "is_clean_day", "is_stockout_day", "stockout_hours", "hours_sale"]].copy()
    df = df.sort_values(["series_id", "dt"]).reset_index(drop=True)

    # Sample which clean training days are masked.
    candidate = ((df.split == "train") & (df.is_clean_day == 1)).to_numpy()
    prob = np.zeros(len(df))
    cand = df[candidate]
    prob[candidate] = mask_probabilities(cand.sale_amount, cand.series_id, cfg.mask_rate, cfg.demand_power)
    masked = rng.random(len(df)) < prob

    # Draw simulated stockout start hours from the real training distribution.
    start_dist = stockout_start_distribution(panel)
    start = np.full(len(df), -1, dtype=np.int16)
    start[masked] = rng.choice(start_dist.index.to_numpy(), size=masked.sum(), p=start_dist.to_numpy())

    true = df.sale_amount.to_numpy()
    hs = df.hours_sale.to_numpy()
    censored = true.copy()
    idx = np.flatnonzero(masked)
    censored[idx] = [censor_day(hs[i], start[i]) for i in idx]

    # Matched noise: same absolute error as censoring, with symmetric +/- sign.
    # Result is always non-negative because distortion <= true.
    distortion = np.abs(true - censored)
    sign = rng.choice([-1.0, 1.0], size=len(df))
    noised = true + sign * distortion

    df["is_masked"] = masked.astype("int8")
    df["mask_prob"] = prob.astype("float32")
    df["sim_stockout_hour"] = start
    df["sales_true"] = true
    df["sales_A"] = true
    df["sales_B"] = censored
    df["sales_C"] = noised
    return df.drop(columns=["hours_sale"]), start_dist
