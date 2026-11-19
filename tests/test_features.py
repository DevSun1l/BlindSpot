"""Unit tests for leakage-safe feature generation and train/eval split flags."""
import numpy as np
import pandas as pd

from blindspot.features import FEATURE_GROUPS, FEATURES, FORBIDDEN, build_features


def _toy(n_series=3, n_train=40, n_eval=7, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_train + n_eval, freq="D")
    rows, cov = [], []
    for s in range(n_series):
        for d in range(n_train + n_eval):
            true = float(rng.poisson(3))
            masked = d < n_train and d % 4 == 0
            rows.append(dict(series_id=s, store_id=s, product_id=s, dt=dates[d], day_idx=d,
                             split="train" if d < n_train else "eval", sale_amount=true,
                             is_clean_day=int(d % 9 != 0), is_stockout_day=int(d % 9 == 0),
                             stockout_hours=0, is_masked=int(masked), mask_prob=0.0, sim_stockout_hour=-1,
                             sales_true=true, sales_A=true, sales_B=true * (0.5 if masked else 1),
                             sales_C=true * (1.5 if masked else 1)))
            cov.append(dict(series_id=s, dt=dates[d], discount=1.0, activity_flag=0, holiday_flag=0, precpt=0.0,
                            avg_temperature=20.0, avg_humidity=50.0, avg_wind_level=1.0, city_id=0,
                            first_category_id=0, second_category_id=0, third_category_id=0, sales_group="low"))
    return pd.DataFrame(rows), pd.DataFrame(cov)


def test_no_forbidden_features():
    assert not FORBIDDEN & set(FEATURES)


def test_changing_day_t_sales_does_not_change_day_t_features():
    cond, cov = _toy()
    base = build_features(cond, cov, "A")
    i = cond.index[(cond.series_id == 1) & (cond.day_idx == 25)][0]
    cond2 = cond.copy()
    cond2.loc[i, "sales_A"] += 100
    new = build_features(cond2, cov, "A")
    hist = FEATURE_GROUPS["recent_sales"]
    # Modifying day t must not affect day t's own features, but must appear in day t+1's lag_1.
    row = (base.series_id == 1) & (base.day_idx == 25)
    pd.testing.assert_frame_equal(base.loc[row, hist], new.loc[row, hist])
    nxt = (base.series_id == 1) & (base.day_idx == 26)
    assert new.loc[nxt, "lag_1"].iloc[0] == base.loc[nxt, "lag_1"].iloc[0] + 100


def test_condition_history_on_train_and_recorded_history_on_test():
    cond, cov = _toy()
    a, b = build_features(cond, cov, "A"), build_features(cond, cov, "B")
    te = a.is_test == 1
    hist = FEATURE_GROUPS["recent_sales"]
    pd.testing.assert_frame_equal(a.loc[te, hist], b.loc[te, hist])
    tr = (a.split == "train") & a.lag_1.notna()
    assert (b.loc[tr, "lag_1"] < a.loc[tr, "lag_1"]).any()


def test_targets_and_flags():
    cond, cov = _toy()
    b = build_features(cond, cov, "B")
    tt = b.is_train_target == 1
    assert (b.loc[tt, "is_clean_day"] == 1).all() and (b.loc[tt, "day_idx"] >= 14).all()
    assert (b.loc[tt, "split"] == "train").all()
    assert (b.loc[b.is_test == 1, "target"] == b.loc[b.is_test == 1, "sales_true"]).all()
    assert (b.loc[b.is_valid == 1, "day_idx"].min() > b.loc[tt & (b.is_valid == 0), "day_idx"].max())
