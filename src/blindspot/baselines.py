"""Forecasting baselines: next-day heuristics, Croston-SBA, and week-ahead SSA.

Next-day baselines use history up to t-1 (recorded sales at test time):
  naive           y_hat(t) = y(t-1)
  seasonal_naive  y_hat(t) = y(t-7)
  moving_avg_7    y_hat(t) = mean(y(t-7..t-1))
  croston_sba     Croston with Syntetos-Boylan bias correction (alpha = 0.1)

Week-ahead baseline (uses only the 90 training days per condition):
  ssa             Similar Scenario Average, ported from frn-50k-baseline
                  (demand_forecasting/SSA/ssa_forecasting.py, Apache 2.0, Dingdong-Inc)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Holiday overrides matching the reference frn-50k-baseline SSA implementation.
HOLIDAY_OVERRIDE_DATES = ["2024-05-02", "2024-05-03", "2024-05-04"]


def croston_sba(y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Compute one-step-ahead Croston-SBA forecasts for a (n_series, n_days) array.

    The forecast for day t uses only observations strictly before t; day 0 and
    any days prior to the first non-zero sale are NaN.
    """
    n, T = y.shape
    z = np.full(n, np.nan)  # Smoothed non-zero demand size
    p = np.full(n, np.nan)  # Smoothed inter-demand interval
    q = np.ones(n)          # Periods elapsed since last non-zero demand
    out = np.full((n, T), np.nan)
    for t in range(T):
        # Syntetos-Boylan approximation applies the (1 - alpha / 2) debiasing factor.
        out[:, t] = (1 - alpha / 2) * z / p
        nz = y[:, t] > 0
        first = nz & np.isnan(z)
        upd = nz & ~first
        # Initialize on first observed sale; exponentially smooth subsequent sales.
        z[first], p[first] = y[first, t], q[first]
        z[upd] = z[upd] + alpha * (y[upd, t] - z[upd])
        p[upd] = p[upd] + alpha * (q[upd] - p[upd])
        q = np.where(nz, 1, q + 1)
    return out


def next_day_baselines(feats: pd.DataFrame) -> pd.DataFrame:
    """Append naive, seasonal_naive, moving_avg_7, and croston_sba columns to a feature table
    sorted by (series_id, dt) with equal length per series."""
    f = feats.sort_values(["series_id", "dt"]).reset_index(drop=True).copy()
    f["naive"] = f["lag_1"]
    f["seasonal_naive"] = f["lag_7"]
    f["moving_avg_7"] = f["roll_mean_7"]
    # Run Croston-SBA over recorded sales so test-time history matches the feature pipeline.
    hist = f["sales_true"].to_numpy()
    n_days = f.groupby("series_id").size().iloc[0]
    f["croston_sba"] = croston_sba(hist.reshape(-1, n_days)).reshape(-1)
    return f


def ssa_forecast(df: pd.DataFrame, target_col: str, encoder_len: int = 90) -> np.ndarray:
    """Port of the frn-50k-baseline Similar Scenario Average (SSA) forecaster.

    Expects 97 consecutive daily rows per series (90 train + 7 test), sorted by
    (series_id, dt), with columns dt, discount, holiday_flag, precpt, and `target_col`.
    Uses only the first `encoder_len` days of `target_col` and returns predictions
    for the 7 test days as a flat array of length (n_series * 7).
    """
    d = df[["dt", "discount", "holiday_flag", "precpt", target_col]].copy()
    dt = pd.to_datetime(d["dt"])
    assert len(d) % 97 == 0, "SSA port expects 97 days per series"
    d["time_idx"] = (dt - dt.min()).dt.days
    d["dow"] = dt.dt.dayofweek
    # Normalize holiday/weekend day-of-week buckets and heavy-rain indicator per reference SSA rules.
    d.loc[dt.dt.strftime("%Y-%m-%d").isin(HOLIDAY_OVERRIDE_DATES), "holiday_flag"] = 0
    d.loc[d["holiday_flag"] == 1, "dow"] = 5
    d.loc[(d["holiday_flag"] == 0) & (d["dow"].isin([5, 6])), "dow"] = 0
    d["is_precpt"] = (d["precpt"] > 3.5).astype(int)

    x_cont = d[["time_idx", "discount", target_col]].to_numpy(dtype=float).reshape(-1, 97, 3)
    x_cat = d[["dow", "holiday_flag", "is_precpt"]].to_numpy().reshape(-1, 97, 3)
    L, holiday_label = encoder_len, 5

    # Compute pairwise similarity between each forecast day (decoder) and historical day (encoder)
    # based on recency, discount proximity, holiday/weekday match, and precipitation match.
    date_distance = 18 - x_cont[:, :L, 0][:, None, :] / 5
    discount_distance = 1 - np.abs(x_cont[:, L:, 1][..., None] - x_cont[:, :L, 1][:, None, :])
    decoder_dow = x_cat[:, L:, 0][..., None]
    encoder_dow = x_cat[:, :L, 0][:, None, :]
    dow_distance = decoder_dow - encoder_dow
    decoder_precpt = x_cat[:, L:, 2][..., None]
    encoder_precpt = x_cat[:, :L, 2][:, None, :]
    precpt_distance = decoder_precpt - encoder_precpt
    date_distance = np.where((decoder_dow == holiday_label).astype(int) - (encoder_dow == holiday_label).astype(int) == 0,
                             date_distance - 3, date_distance)
    date_distance = np.where((decoder_dow != holiday_label) & (encoder_dow == holiday_label), date_distance + 3, date_distance)
    date_distance = np.where(dow_distance == 0, date_distance - 3, date_distance)
    date_distance = np.where((precpt_distance == 0) & (decoder_precpt == 1), date_distance - 3, date_distance)
    date_distance = np.where(precpt_distance != 0, date_distance + 3, date_distance)
    date_distance = np.clip(date_distance, 1, 18)
    weight = (18 - date_distance) * discount_distance ** 2
    weight = np.exp(weight) / np.exp(weight).sum(axis=-1, keepdims=True)

    # Estimate cross-series day-of-week seasonality ratios and compute weighted historical average.
    target = x_cont[:, :L, 2:3]
    index = encoder_dow.squeeze(1)[..., None] == np.arange(6)
    mean_sale = np.nanmean(np.where(index, target, np.nan), axis=1) + 0.001
    season_ratio = np.nanmedian(mean_sale[..., None] / mean_sale[:, None, ...], axis=0)
    ratio = season_ratio[decoder_dow, encoder_dow]
    pred = (weight * ratio) @ target
    return pred.reshape(-1)
