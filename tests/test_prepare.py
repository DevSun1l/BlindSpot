"""Unit tests for discount cleaning and sampling, plus integration checks on the prepared panel."""
import pandas as pd
import pytest

from blindspot.prepare import PROCESSED_DIR, clean_discount, draw_sample, load_panel, nested_samples, resize_sample


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


def _tab(n=400):
    return pd.DataFrame({
        "series_id": range(n), "eligible": [i % 10 != 0 for i in range(n)],
        "first_category_id": [i % 3 for i in range(n)],
        "sales_group": ["low" if i % 4 else "high" for i in range(n)],
    })


def test_resize_sample_is_nested_and_stratified():
    tab = _tab()
    base = draw_sample(tab, n=100, seed=42)
    small, same, big = (resize_sample(tab, base, n) for n in (50, 100, 200))
    assert (len(small), len(same), len(big)) == (50, 100, 200)
    assert small.series_id.isin(base.series_id).all() and base.series_id.isin(big.series_id).all()
    assert same.series_id.tolist() == sorted(base.series_id)
    assert big.series_id.is_unique and big.eligible.all()
    pool_low = (tab[tab.eligible].sales_group == "low").mean()
    for s in (small, big):
        assert abs((s.sales_group == "low").mean() - pool_low) < 0.03


def test_resize_sample_too_large():
    tab = _tab()
    with pytest.raises(ValueError):
        resize_sample(tab, draw_sample(tab, n=100), n=1000)


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


def test_nested_samples_chain():
    tab = _tab()
    base = draw_sample(tab, n=100, seed=42)
    s = nested_samples(tab, base, [30, 60, 100, 150, 250])
    sizes = sorted(s)
    assert [len(s[n]) for n in sizes] == sizes
    for small, big in zip(sizes, sizes[1:]):
        assert s[small].series_id.isin(s[big].series_id).all()
    assert s[100].series_id.tolist() == sorted(base.series_id)
