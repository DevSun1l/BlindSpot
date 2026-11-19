"""Unit tests for Streamlit data helpers and end-to-end headless app smoke test."""
import numpy as np
import pandas as pd
import pytest

from blindspot.app_data import cost_curve, noise_floor_from_report, order_table, safety_for, top_drivers
from blindspot.data import PROJECT_ROOT
from blindspot.features import FEATURE_GROUPS, FEATURES
from blindspot.prepare import PROCESSED_DIR

APP = PROJECT_ROOT / "app" / "streamlit_app.py"
NEEDED = [PROCESSED_DIR / "features_A.parquet", PROCESSED_DIR / "validation_errors.parquet",
          PROJECT_ROOT / "outputs" / "models" / "lgb_A.txt"]


def test_order_table_and_cost_curve():
    val = pd.DataFrame({"condition": ["A"] * 4 + ["B"] * 4, "group": ["high"] * 8,
                        "err_own": [0, 1, 2, 3, -1, 0, 1, 2]})
    # Critical ratio 0.5 corresponds to the median validation residual.
    s = safety_for(val, cu=1.0, co=1.0)
    assert np.isclose(s["A"]["high"], 1.5) and np.isclose(s["B"]["high"], 0.5)
    t = order_table({"A": 4.0, "B": -3.0}, "high", s, 1.0, 1.0, true_demand=5.0)
    assert np.allclose(t["recommended order"], [5.5, 0.0])  # Orders are clipped at 0
    assert np.allclose(t["cost"], [0.5, 5.0])
    assert "cost" not in order_table({"A": 4.0}, "high", s, 1.0, 1.0, None).columns
    cc = cost_curve({"A": 2.0}, 4.0, cu=1.0, co=0.5, n=5)
    assert np.allclose(cc.A, [1.0, 0.5, 0.0, 1.0, 2.0])


def test_top_drivers():
    sv = pd.DataFrame(0.0, index=FEATURES, columns=list("ABC"))
    sv.loc["discount"] = [0.5, 0.1, 0.4]
    sv.loc["lag_1"] = [-2.0, -1.0, -2.0]
    sv.loc["weekday"] = [0.0, 3.0, 0.0]
    top = top_drivers(sv, k=3)
    assert set(top.index) == {"discount", "lag_1", "weekday"}
    # Sorted ascending by |A| so the largest driver renders at the top of a horizontal bar chart.
    assert top.index[-1] == "lag_1"
    g = top_drivers(sv, k=5, grouped=True)
    assert set(g.index) <= set(FEATURE_GROUPS) and np.isclose(g.loc["promotion", "A"], 0.5)


def test_noise_floor_parse(tmp_path):
    p = tmp_path / "r.md"
    p.write_text("| A vs A' (mean of 3) | 0.944 (0.941-0.947) | 0.88 | 70% |", encoding="utf-8")
    assert noise_floor_from_report(p) == 0.944
    assert noise_floor_from_report(tmp_path / "missing.md") is None


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="pipeline outputs not present")
def test_app_runs_end_to_end():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, at.exception
    assert "Store" in at.title[0].value

    day = next(s for s in at.selectbox if s.label.startswith("Forecast day"))
    stock = [i for i, o in enumerate(day.options) if "stockout" in o]
    if stock:
        at = day.select_index(stock[0]).run()
        assert not at.exception, at.exception
        assert any("real stockout day" in i.value for i in at.info)

    # Untouched page: the actual plan is used.
    assert not any("differs from what actually happened" in c.value for c in at.caption)
    # Switching to a custom plan without moving the slider must still use the chosen (rounded) values,
    # not silently fall back to the actual plan.
    at = next(t for t in at.toggle if t.label == "Try a different plan").set_value(True).run()
    assert not at.exception, at.exception
    assert any("differs from what actually happened" in c.value for c in at.caption)
    act = next(t for t in at.toggle if t.label.startswith("Promotion event"))
    at = act.set_value(not act.value).run()
    assert not at.exception, at.exception

    prod = next(s for s in at.selectbox if s.label == "Product")
    low = [i for i, o in enumerate(prod.options) if "low demand" in o] or [0]
    at = prod.select_index(low[0]).run()
    assert not at.exception, at.exception
