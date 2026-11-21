"""Unit tests for the Step 17 censoring approaches: imputation, censored loss and approach feature tables."""
import json

import numpy as np
import pandas as pd
import pytest

from blindspot.data import PROJECT_ROOT
from blindspot.mitigation import (MIN_SHARE, ShiftedBooster, approach_features, censored_objective, hourly_share,
                                  impute_demand, train_censored)
from blindspot.prepare import PROCESSED_DIR


DATES = pd.date_range("2024-01-01", periods=40, freq="D")


def _toy(n_days=40):
    """Two series; every hour 8-17 sells 1 unit (10/day). Day 30 of series 0 stocks out at 13:00 (5 units kept)."""
    rows, panel = [], []
    hours = np.zeros(24)
    hours[8:18] = 1.0
    for s in (0, 1):
        for d in range(n_days):
            masked = int(s == 0 and d == 30)
            rows.append(dict(series_id=s, dt=DATES[d], split="train",
                             is_clean_day=1, is_masked=masked, sim_stockout_hour=13 if masked else -1,
                             sales_true=10.0, sales_B=5.0 if masked else 10.0))
            panel.append(dict(series_id=s, dt=rows[-1]["dt"], hours_sale=hours))
    return pd.DataFrame(rows), pd.DataFrame(panel)


def test_hourly_share_and_ratio_imputation():
    cond, panel = _toy()
    share = hourly_share(cond, panel)
    assert np.isclose(share.loc[0, 13], 0.5, atol=0.01) and np.isclose(share.loc[0, 24], 1.0)
    imp = impute_demand(cond, panel)
    # 5 units by 13:00, when half the day's sales are usually made -> about 10.
    assert np.isclose(imp[cond.is_masked == 1][0], 10.0, atol=0.3)
    assert np.allclose(imp[cond.is_masked == 0], cond.sales_B[cond.is_masked == 0])   # other days untouched


def test_early_stockout_uses_fill_not_ratio():
    cond, panel = _toy()
    i = cond.index[cond.is_masked == 1][0]
    cond.loc[i, ["sim_stockout_hour", "sales_B"]] = [9, 1.0]     # 10% of the day by 09:00, below MIN_SHARE
    assert MIN_SHARE > 0.1
    imp = impute_demand(cond, panel)[i]
    assert np.isclose(imp, 1.0 + 0.9 * 10.0, atol=0.3)          # 1 sold + 90% of a usual 10-unit day


def test_censored_objective_one_sided():
    grad, hess = censored_objective(np.array([0, 0, 1, 1]))(np.array([2.0, 0.0, 2.0, 0.0]), _Label([1.0] * 4))
    # Uncensored: plain squared-loss gradient. Censored: only under-forecasting (pred < label) is penalised.
    assert np.allclose(grad, [1.0, -1.0, 0.0, -1.0]) and np.allclose(hess, 1.0)


class _Label:
    def __init__(self, y):
        self.y = np.asarray(y)

    def get_label(self):
        return self.y


def test_shifted_booster_adds_offset():
    class B:
        def predict(self, X, **kw):
            return np.zeros(len(X))
    assert np.allclose(ShiftedBooster(B(), 2.5).predict(np.zeros((3, 1))), 2.5)


NEEDED = [PROCESSED_DIR / "panel.parquet", PROCESSED_DIR / "sample_series.parquet",
          PROJECT_ROOT / "outputs" / "models" / "lgb_config.json"]


@pytest.fixture(scope="module")
def small():
    if not all(p.exists() for p in NEEDED):
        pytest.skip("pipeline outputs not present")
    from blindspot.conditions import ConditionConfig, build_conditions
    from blindspot.prepare import load_panel
    panel = load_panel()
    panel = panel[panel.series_id.isin(panel.series_id.drop_duplicates().iloc[:200])].reset_index(drop=True)
    cond, _ = build_conditions(panel, ConditionConfig(seed=42))
    return panel, cond


def test_custom_loss_without_censoring_matches_builtin(small):
    from blindspot.features import build_features
    from blindspot.model import predict, split_rows, train_final
    panel, cond = small
    f = build_features(cond, panel, "A").assign(censored=0)
    _, _, tr, te = split_rows(f)
    p = {**json.loads(NEEDED[2].read_text())["params"], "feature_fraction": 1.0, "bagging_fraction": 1.0}
    a, b = predict(train_final(p, 20, tr), te), predict(train_censored(p, 20, tr), te)
    assert np.abs(a - b).max() < 1e-4


def test_approach_features(small):
    panel, cond = small
    naive, drop = approach_features(cond, panel, "naive"), approach_features(cond, panel, "drop")
    imp, cen = approach_features(cond, panel, "impute"), approach_features(cond, panel, "censored_loss")
    stock = (naive.is_masked == 1) & (naive.split == "train")
    assert naive.target.equals(cen.target)                                   # same censored labels
    assert not drop.loc[stock, "is_train_target"].any() and naive.loc[stock, "is_train_target"].any()
    assert (cen.loc[stock & (cen.is_train_target == 1), "censored"] == 1).all() and naive.censored.eq(0).all()
    assert (imp.target >= naive.target - 1e-5).all()                         # imputation never lowers sales
    assert imp.loc[stock, "target"].sum() > naive.loc[stock, "target"].sum()
    test = naive.is_test == 1                                                # test inputs identical everywhere
    for f in (drop, imp, cen):
        assert np.allclose(f.loc[test, "lag_1"].fillna(-1), naive.loc[test, "lag_1"].fillna(-1))
        assert np.allclose(f.loc[test, "target"], naive.loc[test, "target"])
