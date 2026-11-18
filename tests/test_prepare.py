"""Step 2 tests: discount cleaning (synthetic) and the prepared panel (skipped if not built)."""
import pandas as pd
import pytest

from blindspot.prepare import PROCESSED_DIR, clean_discount, draw_sample, load_panel


def test_clean_discount_rules():
    df = pd.DataFrame({"discount": [0.0, -0.1, 0.5, 1.0, 1.088]})
    out = clean_discount(df)
    assert out["discount"].tolist() == [1.0, 1.0, 0.5, 1.0, 1.0]
    assert out["discount_fixed"].tolist() == [1, 1, 0, 0, 1]


def test_draw_sample_is_reproducible_and_sized():
    tab = pd.DataFrame({
        "series_id": range(100), "eligible": [True] * 90 + [False] * 10,
        "first_category_id": [i % 3 for i in range(100)],
        "sales_group": ["low" if i % 4 else "high" for i in range(100)],
    })
    a, b = draw_sample(tab, n=30, seed=1), draw_sample(tab, n=30, seed=1)
    assert len(a) == 30 and a.series_id.tolist() == b.series_id.tolist()
    assert a.eligible.all()


built = pytest.mark.skipif(not (PROCESSED_DIR / "panel.parquet").exists(),
                           reason="run scripts/step2_prepare_data.py first")


@built
def test_sample_panel():
    p = load_panel(sample_only=True)
    assert p.series_id.nunique() == 5000
    assert p.groupby("series_id").size().eq(97).all()
    assert p.discount.between(0, 1, inclusive="right").all()
    assert "stock_hour6_22_cnt" not in p.columns
    assert ((p.is_clean_day == 1) == (p.stockout_hours == 0)).all()
