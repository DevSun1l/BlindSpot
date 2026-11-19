"""Step 9 tests: what-if guard rails and uplift on small synthetic data."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from blindspot.features import CATEGORICAL, FEATURES
from blindspot.whatif import apply_changes, bootstrap_ratio_ci, discount_curve, observed_activity_ratio, whatif


def _model(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    for c in CATEGORICAL:
        X[c] = pd.Categorical(rng.integers(0, 4, n))
    X["discount"] = rng.uniform(0.5, 1.0, n)
    X["activity_flag"] = rng.integers(0, 2, n)
    y = 5 + X["lag_1"] + 4 * (1 - X["discount"]) + 1.0 * X["activity_flag"] + rng.normal(0, 0.1, n)
    b = lgb.train({"verbosity": -1, "seed": 0, "num_leaves": 15}, lgb.Dataset(X[FEATURES], y, categorical_feature=CATEGORICAL), 200)
    return b, X


@pytest.mark.parametrize("bad", [{"lag_1": 3.0}, {"roll_mean_7": 1.0}, {"store_id": 1}, {"discount": 0.3},
                                 {"discount": 1.2}, {"activity_flag": 0.5}, {"activity_flag": 2}])
def test_guard_rails_reject(bad):
    X = pd.DataFrame({f: [1.0] for f in FEATURES})
    with pytest.raises(ValueError):
        apply_changes(X, bad)


def test_apply_changes_only_touches_levers():
    _, X = _model()
    Z = apply_changes(X, {"discount": 0.8, "activity_flag": 1})
    assert (Z["discount"] == 0.8).all() and (Z["activity_flag"] == 1).all()
    other = [f for f in FEATURES if f not in ("discount", "activity_flag")]
    pd.testing.assert_frame_equal(Z[other], X[other])
    assert not (X["discount"] == 0.8).all()          # original untouched


def test_whatif_recovers_known_effects():
    b, X = _model()
    act = whatif(b, X.iloc[:500], {"activity_flag": 0}, {"activity_flag": 1})
    assert abs(act.uplift.mean() - 1.0) < 0.15
    np.testing.assert_allclose(act.uplift, act.scenario - act.base)
    disc = whatif(b, X.iloc[:500], {"discount": 1.0}, {"discount": 0.8})
    assert abs(disc.uplift.mean() - 0.8) < 0.15
    cv = discount_curve(b, X.iloc[:500], grid=[1.0, 0.8, 0.6])
    assert cv[1.0] < cv[0.8] < cv[0.6]


def test_ratio_ci_and_observed_ratio():
    rng = np.random.default_rng(0)
    den = rng.uniform(1, 2, 1000)
    lo, hi = bootstrap_ratio_ci(0.6 * den, den, np.arange(1000) % 40)
    assert lo <= 0.6 <= hi and hi - lo < 1e-9 + 0.01
    df = pd.DataFrame({"series_id": [1, 1, 2, 2, 3], "activity_flag": [0, 1, 0, 1, 0], "target": [1, 1.5, 2, 2.6, 4]})
    assert np.isclose(observed_activity_ratio(df), 1.4)        # median of 1.5 and 1.3; series 3 has no activity day
