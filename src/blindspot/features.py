"""Leakage-safe feature engineering for next-day forecasting under conditions A, B, and C.

Design:
- Sales-history features for day t use only observations up to day t-1.
- Training rows derive history from the condition's own sales column.
- Evaluation rows derive history from recorded sales (condition A) so all models
  receive identical test-time inputs.
- Promotion, calendar, and weather covariates for day t are treated as known ahead of time.
- Stockout and masking indicators are excluded from model features.
- Training targets are restricted to clean training days after the 14-day warmup window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 2, 7, 14]
WARMUP_DAYS = 14
VALID_DAYS = 14  # Trailing training days reserved for validation / early stopping

FEATURE_GROUPS: dict[str, list[str]] = {
    "recent_sales": [f"lag_{k}" for k in LAGS] + ["roll_mean_7", "roll_mean_14", "roll_std_7"],
    "calendar": ["weekday", "is_weekend", "holiday_flag"],
    "promotion": ["discount", "activity_flag"],
    "weather": ["precpt", "avg_temperature", "avg_humidity", "avg_wind_level"],
    "identity": ["store_id", "product_id", "city_id", "first_category_id", "second_category_id", "third_category_id"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
CATEGORICAL = FEATURE_GROUPS["identity"] + ["weekday"]
FORBIDDEN = {"stockout_hours", "is_stockout_day", "is_clean_day", "hours_stock_status", "stock_hour6_22_cnt",
             "is_masked", "mask_prob", "sim_stockout_hour"}
assert not FORBIDDEN & set(FEATURES)

COVARIATES = ["discount", "activity_flag", "holiday_flag", "precpt", "avg_temperature", "avg_humidity",
              "avg_wind_level", "city_id", "first_category_id", "second_category_id", "third_category_id",
              "sales_group"]


def history_features(df: pd.DataFrame, sales_col: str) -> pd.DataFrame:
    """Compute lagged and rolling statistics from `sales_col` (requires consecutive daily rows per series)."""
    g = df.groupby("series_id", sort=False)[sales_col]
    # Shift by 1 day before rolling aggregations so day t never leaks into its own features.
    past = g.shift(1)
    gp = past.groupby(df["series_id"], sort=False)
    out = pd.DataFrame(index=df.index)
    for k in LAGS:
        out[f"lag_{k}"] = g.shift(k)
    out["roll_mean_7"] = gp.transform(lambda s: s.rolling(7, min_periods=7).mean())
    out["roll_mean_14"] = gp.transform(lambda s: s.rolling(14, min_periods=14).mean())
    out["roll_std_7"] = gp.transform(lambda s: s.rolling(7, min_periods=7).std())
    return out.astype("float32")


def build_features(cond: pd.DataFrame, panel: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Build the feature and target table for condition 'A', 'B', or 'C'."""
    sales_col = f"sales_{condition}"
    df = cond.sort_values(["series_id", "dt"]).reset_index(drop=True)
    cov = panel[["series_id", "dt"] + COVARIATES]
    df = df.merge(cov, on=["series_id", "dt"], how="left", validate="one_to_one")

    # Use condition-specific history on train rows and recorded history (sales_A) on eval rows.
    own = history_features(df, sales_col)
    rec = history_features(df, "sales_A")
    is_eval = (df["split"] == "eval").to_numpy()
    hist = own.copy()
    hist.loc[is_eval] = rec.loc[is_eval]

    feats = pd.concat([df, hist], axis=1)
    feats["weekday"] = feats["dt"].dt.dayofweek.astype("int8")
    feats["is_weekend"] = (feats["weekday"] >= 5).astype("int8")
    feats["condition"] = condition
    feats["target"] = np.where(is_eval, feats["sales_true"], feats[sales_col]).astype("float32")

    # Define training, validation, and test masks (test rows are later scored on clean days only).
    train = feats["split"] == "train"
    last_train = feats.loc[train, "day_idx"].max()
    feats["is_train_target"] = (train & (feats["is_clean_day"] == 1) & (feats["day_idx"] >= WARMUP_DAYS)).astype("int8")
    feats["is_valid"] = (feats["is_train_target"].astype(bool) & (feats["day_idx"] > last_train - VALID_DAYS)).astype("int8")
    feats["is_test"] = is_eval.astype("int8")

    for c in CATEGORICAL:
        feats[c] = feats[c].astype("category")
    keep = ["series_id", "dt", "day_idx", "split", "condition", "sales_group", "is_clean_day", "is_masked",
            "sales_true", "target", "is_train_target", "is_valid", "is_test"] + FEATURES
    return feats[keep]
