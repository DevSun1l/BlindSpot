"""Step 4: leakage-safe features for next-day forecasting, per condition (A, B, C).

Rules
- Every sales-history feature for day t uses sales up to day t-1 only (shift by 1).
- TRAIN rows: history comes from the condition's own sales (so B's lags are censored).
- TEST (eval) rows: history comes from the recorded sales (= condition A) in every condition,
  so all three models see identical test inputs and differ only in what they learned.
- Promotion, calendar and weather values for day t are treated as known in advance.
- No stockout information is a feature.

Training targets: clean train days only (masked days carry the condition's value). Real stockout
days are never targets in any condition, but their recorded sales still appear in the lags.
The first 14 days are dropped as targets because their 14-day history is incomplete.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 2, 7, 14]
WARMUP_DAYS = 14
VALID_DAYS = 14          # last 14 train days -> validation (early stopping)

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
    """Lag and rolling features from `sales_col`; df must be sorted by series_id, dt (consecutive days)."""
    g = df.groupby("series_id", sort=False)[sales_col]
    past = g.shift(1)                                    # sales up to t-1
    gp = past.groupby(df["series_id"], sort=False)
    out = pd.DataFrame(index=df.index)
    for k in LAGS:
        out[f"lag_{k}"] = g.shift(k)
    out["roll_mean_7"] = gp.transform(lambda s: s.rolling(7, min_periods=7).mean())
    out["roll_mean_14"] = gp.transform(lambda s: s.rolling(14, min_periods=14).mean())
    out["roll_std_7"] = gp.transform(lambda s: s.rolling(7, min_periods=7).std())
    return out.astype("float32")


def build_features(cond: pd.DataFrame, panel: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Feature table for one condition ('A', 'B' or 'C')."""
    sales_col = f"sales_{condition}"
    df = cond.sort_values(["series_id", "dt"]).reset_index(drop=True)
    cov = panel[["series_id", "dt"] + COVARIATES]
    df = df.merge(cov, on=["series_id", "dt"], how="left", validate="one_to_one")

    own = history_features(df, sales_col)
    rec = history_features(df, "sales_A")                # recorded history, for test rows
    is_eval = (df["split"] == "eval").to_numpy()
    hist = own.copy()
    hist.loc[is_eval] = rec.loc[is_eval]

    feats = pd.concat([df, hist], axis=1)
    feats["weekday"] = feats["dt"].dt.dayofweek.astype("int8")
    feats["is_weekend"] = (feats["weekday"] >= 5).astype("int8")
    feats["condition"] = condition
    feats["target"] = np.where(is_eval, feats["sales_true"], feats[sales_col]).astype("float32")

    train = feats["split"] == "train"
    last_train = feats.loc[train, "day_idx"].max()
    feats["is_train_target"] = (train & (feats["is_clean_day"] == 1) & (feats["day_idx"] >= WARMUP_DAYS)).astype("int8")
    feats["is_valid"] = (feats["is_train_target"].astype(bool) & (feats["day_idx"] > last_train - VALID_DAYS)).astype("int8")
    feats["is_test"] = is_eval.astype("int8")        # scored only where is_clean_day == 1

    for c in CATEGORICAL:
        feats[c] = feats[c].astype("category")
    keep = ["series_id", "dt", "day_idx", "split", "condition", "sales_group", "is_clean_day", "is_masked",
            "sales_true", "target", "is_train_target", "is_valid", "is_test"] + FEATURES
    return feats[keep]
