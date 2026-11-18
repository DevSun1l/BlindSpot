"""Step 1 tests on the cached dataset. Skipped if data/raw/ has not been downloaded yet."""
import numpy as np
import pytest

from blindspot.data import BUSINESS_HOURS, EXPECTED_COLUMNS, RAW_DIR, load_split

pytestmark = pytest.mark.skipif(not (RAW_DIR / "eval.parquet").exists(),
                                reason="run scripts/step1_check_data.py first")


@pytest.fixture(scope="module")
def evl():
    return load_split("eval")


def test_columns(evl):
    assert set(evl.columns) == set(EXPECTED_COLUMNS)


def test_shape(evl):
    assert len(evl) == 350_000
    assert evl.groupby(["store_id", "product_id"]).size().eq(7).all()


def test_stockout_encoding(evl):
    st = np.stack(evl["hours_stock_status"].to_numpy())
    assert ((st[:, BUSINESS_HOURS] == 1).sum(axis=1) == evl["stock_hour6_22_cnt"].to_numpy()).all()


def test_hourly_sums_to_daily(evl):
    hs = np.stack(evl["hours_sale"].to_numpy())
    assert np.allclose(hs.sum(axis=1), evl["sale_amount"].to_numpy(), rtol=1e-3, atol=1e-6)
