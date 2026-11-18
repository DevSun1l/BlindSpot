"""Step 2: cleaning and preparation of the FreshRetailNet-50K panel.

Output: one long panel (train + eval days) with stockout information kept in separate
columns, a clean-day flag, a sales-volume group, and a reproducible study sample of series.
Stockout columns are NOT model features; they are used only to build and evaluate conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from blindspot.data import KEYS, PROJECT_ROOT, load_split

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SERIES = ["store_id", "product_id"]

# Eligibility for the study (series must have enough clean days to mask and to evaluate)
MIN_CLEAN_TRAIN_DAYS = 30
MIN_CLEAN_EVAL_DAYS = 1
# Low/high sales split, same threshold as frn-50k-baseline (mean daily train sales < 1 = low)
LOW_SALES_THRESHOLD = 1.0
DEFAULT_SAMPLE_SIZE = 5_000
SEED = 42


def clean_discount(df: pd.DataFrame) -> pd.DataFrame:
    """discount <= 0 is invalid -> 1.0 (flagged); discount > 1 is clipped to 1.0 (flagged)."""
    df = df.copy()
    invalid = df["discount"] <= 0
    above = df["discount"] > 1
    df["discount_fixed"] = (invalid | above).astype("int8")
    df.loc[invalid, "discount"] = 1.0
    df.loc[above, "discount"] = 1.0
    return df


def build_panel() -> pd.DataFrame:
    train = load_split("train").assign(split="train")
    evl = load_split("eval").assign(split="eval")
    df = pd.concat([train, evl], ignore_index=True).sort_values(KEYS).reset_index(drop=True)

    df = clean_discount(df)

    # Stockout information, kept separately (never a model feature)
    df["stockout_hours"] = df["stock_hour6_22_cnt"].astype("int8")
    df["is_stockout_day"] = (df["stockout_hours"] > 0).astype("int8")
    df["is_clean_day"] = (df["stockout_hours"] == 0).astype("int8")

    # Integer series id + day index for convenience
    df["series_id"] = df.groupby(SERIES, sort=True).ngroup().astype("int32")
    df["day_idx"] = (df["dt"] - df["dt"].min()).dt.days.astype("int16")

    # Sales-volume group from TRAIN days only (leakage-safe)
    psd = df[df.split == "train"].groupby("series_id")["sale_amount"].mean().rename("train_mean_sale")
    df = df.merge(psd, on="series_id", how="left")
    df["sales_group"] = np.where(df["train_mean_sale"] < LOW_SALES_THRESHOLD, "low", "high")

    # Compact dtypes
    for c in ["holiday_flag", "activity_flag"]:
        df[c] = df[c].astype("int8")
    df["split"] = df["split"].astype("category")
    df["sales_group"] = df["sales_group"].astype("category")
    return df.drop(columns=["stock_hour6_22_cnt"])


def series_table(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("series_id")
    tab = pd.DataFrame({
        "store_id": g["store_id"].first(),
        "product_id": g["product_id"].first(),
        "city_id": g["city_id"].first(),
        "first_category_id": g["first_category_id"].first(),
        "sales_group": g["sales_group"].first(),
        "train_mean_sale": g["train_mean_sale"].first(),
    })
    tr = panel[panel.split == "train"].groupby("series_id")
    ev = panel[panel.split == "eval"].groupby("series_id")
    tab["clean_train_days"] = tr["is_clean_day"].sum()
    tab["clean_eval_days"] = ev["is_clean_day"].sum()
    tab["stockout_rate_train"] = tr["is_stockout_day"].mean()
    tab["promo_rate_train"] = tr["activity_flag"].mean()
    tab["eligible"] = ((tab.clean_train_days >= MIN_CLEAN_TRAIN_DAYS)
                       & (tab.clean_eval_days >= MIN_CLEAN_EVAL_DAYS)
                       & (tab.train_mean_sale > 0))
    return tab.reset_index()


def draw_sample(tab: pd.DataFrame, n: int = DEFAULT_SAMPLE_SIZE, seed: int = SEED) -> pd.DataFrame:
    """Stratified sample of eligible series by (first_category_id, sales_group), proportional."""
    elig = tab[tab.eligible].copy()
    n = min(n, len(elig))
    frac = n / len(elig)
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in elig.groupby(["first_category_id", "sales_group"], observed=True):
        k = int(round(len(grp) * frac))
        if k:
            parts.append(grp.iloc[rng.choice(len(grp), size=min(k, len(grp)), replace=False)])
    sample = pd.concat(parts)
    # Fix rounding so the sample has exactly n series
    if len(sample) > n:
        sample = sample.iloc[rng.choice(len(sample), size=n, replace=False)]
    elif len(sample) < n:
        rest = elig[~elig.series_id.isin(sample.series_id)]
        sample = pd.concat([sample, rest.iloc[rng.choice(len(rest), size=n - len(sample), replace=False)]])
    return sample.sort_values("series_id").reset_index(drop=True)


def load_panel(sample_only: bool = True) -> pd.DataFrame:
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    if sample_only:
        ids = pd.read_parquet(PROCESSED_DIR / "sample_series.parquet")["series_id"]
        panel = panel[panel.series_id.isin(ids)].reset_index(drop=True)
    return panel
