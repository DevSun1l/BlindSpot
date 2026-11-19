"""Unit tests for forecast evaluation metrics, Croston-SBA, and the SSA baseline."""
import numpy as np
import pandas as pd

from blindspot.baselines import croston_sba, ssa_forecast
from blindspot.metrics import bootstrap_wape, evaluate


def test_evaluate_known_values():
    df = pd.DataFrame({"dt": ["d1"] * 2 + ["d2"] * 2, "sales_group": ["low", "high"] * 2,
                       "is_clean_day": [1, 1, 1, 0], "sales_true": [1.0, 3.0, 2.0, 99.0],
                       "p": [2.0, 3.0, 1.0, 0.0]})
    m = evaluate(df, "p").set_index("group")
    # d1: |err|=1, y=4 (WAPE=0.25, WPE=+0.25); d2 clean row: |err|=1, y=2 (WAPE=0.5, WPE=-0.5).
    assert np.isclose(m.loc["all", "wape"], 0.375)
    assert np.isclose(m.loc["all", "wpe"], -0.125)
    assert m.loc["all", "n_days"] == 3  # Stockout row on d2 is excluded


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"store_id": rng.integers(0, 50, 2000), "dt": rng.integers(0, 7, 2000), "is_clean_day": 1,
                       "sales_group": "high", "sales_true": rng.poisson(3, 2000).astype(float) + 1})
    # Daily error levels differ, so pooled WAPE and the protocol's mean-of-daily WAPE differ too.
    df["p"] = df.sales_true * rng.normal(1, 0.1 + 0.08 * df.dt, 2000)
    lo, hi = bootstrap_wape(df, "p")
    point = evaluate(df, "p").set_index("group").loc["all", "wape"]       # the statistic that is reported
    assert lo < point < hi


def test_croston_sba_constant_demand_and_no_lookahead():
    y = np.full((1, 30), 4.0)
    f = croston_sba(y, alpha=0.1)
    assert np.isnan(f[0, 0])
    # Constant demand with inter-arrival 1 converges to demand * (1 - alpha/2).
    assert np.allclose(f[0, 1:], 4.0 * (1 - 0.05))
    y2 = y.copy()
    y2[0, 20] = 100
    f2 = croston_sba(y2)
    assert np.allclose(f2[0, :21], f[0, :21], equal_nan=True) and f2[0, 21] > f[0, 21]


def test_ssa_shape_and_scale():
    dates = pd.date_range("2024-03-28", periods=97, freq="D")
    rows = []
    for s, level in enumerate([1.0, 5.0]):
        for d in dates:
            rows.append(dict(series_id=s, dt=d, discount=1.0, holiday_flag=int(d.dayofweek >= 5), precpt=0.0, y=level))
    df = pd.DataFrame(rows)
    pred = ssa_forecast(df, "y")
    assert pred.shape == (14,)
    assert np.allclose(pred[:7], 1.0, rtol=0.05) and np.allclose(pred[7:], 5.0, rtol=0.05)
