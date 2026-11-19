"""Step 10 tests: newsvendor outcomes, safety quantiles and order rule."""
import numpy as np
import pandas as pd

from blindspot.inventory import (costs_for_cr, critical_ratio, order_quantity, outcomes, safety_quantiles,
                                 summarize)


def test_outcomes_hand_computed():
    o = outcomes(np.array([5.0, 3.0, 4.0]), np.array([4.0, 5.0, 4.0]), cu=1.0, co=0.5)
    assert o.short.tolist() == [0, 2, 0] and o.waste.tolist() == [1, 0, 0]
    assert o.stockout.tolist() == [0, 1, 0]
    assert np.allclose(o.cost, [0.5, 2.0, 0.0])
    s = summarize(o)
    assert np.isclose(s["fill_rate"], 11 / 13) and np.isclose(s["cost_per_demand"], 2.5 / 13)


def test_critical_ratio_roundtrip():
    assert np.isclose(critical_ratio(1.0, 0.5), 2 / 3)
    cu, co = costs_for_cr(0.8)
    assert np.isclose(critical_ratio(cu, co), 0.8)


def test_safety_and_order_rule():
    resid = np.r_[np.arange(100.0), np.arange(100.0) * 10]
    groups = np.array(["low"] * 100 + ["high"] * 100)
    sq = safety_quantiles(resid, groups, 0.9)
    assert np.isclose(sq["low"], np.quantile(np.arange(100.0), 0.9))
    assert np.isclose(sq["high"], 10 * sq["low"])
    q = order_quantity(np.array([1.0, -5000.0]), np.array(["low", "high"]), sq)
    assert np.isclose(q[0], 1 + sq["low"]) and q[1] == 0          # never negative


def test_newsvendor_quantile_is_cost_optimal():
    """With a perfect-calibration forecast, the CR-quantile safety beats nearby alternatives."""
    rng = np.random.default_rng(0)
    D = rng.gamma(2.0, 1.0, 20000)
    f = np.full_like(D, D.mean())
    cr = 0.8
    cu, co = costs_for_cr(cr)
    g = np.array(["all"] * len(D))
    best = safety_quantiles(D - f, g, cr)["all"]
    cost = lambda s: outcomes(f + s, D, cu, co).cost.mean()  # noqa: E731
    assert cost(best) <= min(cost(best - 0.3), cost(best + 0.3))
    assert isinstance(summarize(outcomes(f, D)), dict) and pd.notna(cost(best))
