"""Cleaning, panel assembly, and stratified series sampling for FreshRetailNet-50K.

Builds a single train+eval panel with stockout indicators stored in separate
columns (excluded from model features and used only for condition construction
and evaluation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from blindspot.data import KEYS, PROJECT_ROOT, load_split

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SERIES = ["store_id", "product_id"]

# Minimum clean-day counts required for a series to support masking and evaluation.
MIN_CLEAN_TRAIN_DAYS = 30
MIN_CLEAN_EVAL_DAYS = 1
# Threshold separating low- and high-volume series (mean daily train sales < 1.0).
LOW_SALES_THRESHOLD = 1.0
DEFAULT_SAMPLE_SIZE = 5_000
SEED = 42


def clean_discount(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize out-of-range discounts (<= 0 or > 1) to 1.0 and record a `discount_fixed` flag."""
    df = df.copy()
    invalid = df["discount"] <= 0
    above = df["discount"] > 1
    df["discount_fixed"] = (invalid | above).astype("int8")
    df.loc[invalid, "discount"] = 1.0
    df.loc[above, "discount"] = 1.0
    return df


def build_panel() -> pd.DataFrame:
    """Combine train and eval splits into a cleaned panel with stockout flags and volume groups."""
    train = load_split("train").assign(split="train")
    evl = load_split("eval").assign(split="eval")
    df = pd.concat([train, evl], ignore_index=True).sort_values(KEYS).reset_index(drop=True)

    df = clean_discount(df)

    # Keep stockout metadata separate from modeling features.
    df["stockout_hours"] = df["stock_hour6_22_cnt"].astype("int8")
    df["is_stockout_day"] = (df["stockout_hours"] > 0).astype("int8")
    df["is_clean_day"] = (df["stockout_hours"] == 0).astype("int8")

    # Dense integer identifiers for series and consecutive day offsets.
    df["series_id"] = df.groupby(SERIES, sort=True).ngroup().astype("int32")
    df["day_idx"] = (df["dt"] - df["dt"].min()).dt.days.astype("int16")

    # Compute sales-volume group from training days only to avoid eval leakage.
    psd = df[df.split == "train"].groupby("series_id")["sale_amount"].mean().rename("train_mean_sale")
    df = df.merge(psd, on="series_id", how="left")
    df["sales_group"] = np.where(df["train_mean_sale"] < LOW_SALES_THRESHOLD, "low", "high")

    for c in ["holiday_flag", "activity_flag"]:
        df[c] = df[c].astype("int8")
    df["split"] = df["split"].astype("category")
    df["sales_group"] = df["sales_group"].astype("category")
    return df.drop(columns=["stock_hour6_22_cnt"])


def series_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-series metadata, stockout/promo rates, and study eligibility."""
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
    """Draw a proportional stratified sample of eligible series by (first_category_id, sales_group)."""
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
    # Adjust for per-stratum rounding so the returned sample has exact size n.
    if len(sample) > n:
        sample = sample.iloc[rng.choice(len(sample), size=n, replace=False)]
    elif len(sample) < n:
        rest = elig[~elig.series_id.isin(sample.series_id)]
        sample = pd.concat([sample, rest.iloc[rng.choice(len(rest), size=n - len(sample), replace=False)]])
    return sample.sort_values("series_id").reset_index(drop=True)


def load_panel(sample_only: bool = True) -> pd.DataFrame:
    """Load the processed panel, optionally filtered to the sampled series."""
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    if sample_only:
        ids = pd.read_parquet(PROCESSED_DIR / "sample_series.parquet")["series_id"]
        panel = panel[panel.series_id.isin(ids)].reset_index(drop=True)
    return panel
