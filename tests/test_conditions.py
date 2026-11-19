"""Unit tests for stockout censoring, demand-linked masking, and matched-noise construction."""
import numpy as np
import pandas as pd

from blindspot.conditions import ConditionConfig, build_conditions, censor_day, mask_probabilities


def test_censor_day_keeps_only_hours_before_stockout():
    hs = np.arange(24, dtype=float)
    assert censor_day(hs, 6) == sum(range(6))
    assert censor_day(hs, 22) == sum(range(22))


def test_mask_probabilities_rate_and_demand_link():
    sales = pd.Series([1, 2, 3, 4, 5, 6, 7, 8.0] * 2)
    sid = pd.Series([0] * 8 + [1] * 8)
    p = mask_probabilities(sales, sid, rate=0.3, power=1.0)
    assert np.allclose(pd.Series(p).groupby(sid).mean(), 0.3)
    # Higher sales percentiles receive higher masking probability when power > 0.
    assert np.all(np.diff(p[:8]) > 0)
    # power=0 reduces to uniform masking at `rate`.
    flat = mask_probabilities(sales, sid, rate=0.3, power=0.0)
    assert np.allclose(flat, 0.3)


def _toy_panel(n_series=20, n_train=60, n_eval=7, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_train + n_eval, freq="D")
    for s in range(n_series):
        for d in range(n_train + n_eval):
            hs = rng.poisson(0.3, 24).astype(float)
            out_ = d % 5 == 0  # Real stockout every 5th day
            status = np.zeros(24, int)
            if out_:
                status[rng.integers(6, 22):] = 1
            rows.append(dict(series_id=s, store_id=s, product_id=0, dt=dates[d],
                             day_idx=d, split="train" if d < n_train else "eval", hours_sale=hs,
                             sale_amount=hs.sum(), hours_stock_status=status,
                             is_clean_day=int(not out_), is_stockout_day=int(out_),
                             stockout_hours=int(status[6:22].sum())))
    return pd.DataFrame(rows)


def test_build_conditions_invariants():
    panel = _toy_panel()
    df, _ = build_conditions(panel, ConditionConfig(mask_rate=0.3, seed=1))
    m = df.is_masked == 1
    assert m.any()
    assert (df.loc[m, "split"] == "train").all() and (df.loc[m, "is_clean_day"] == 1).all()
    assert (df.sales_A == df.sales_true).all()
    assert (df.sales_B <= df.sales_true + 1e-9).all()
    assert np.allclose((df.sales_C - df.sales_true).abs(), (df.sales_B - df.sales_true).abs())
    assert (df.loc[~m, "sales_B"] == df.loc[~m, "sales_true"]).all()
    assert (df.loc[~m, "sales_C"] == df.loc[~m, "sales_true"]).all()


def test_build_conditions_reproducible():
    panel = _toy_panel()
    a, _ = build_conditions(panel, ConditionConfig(seed=3))
    b, _ = build_conditions(panel, ConditionConfig(seed=3))
    pd.testing.assert_frame_equal(a, b)
