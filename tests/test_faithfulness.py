"""Unit tests for deletion/insertion curves, synthetic promo injection, and what-if recovery."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from blindspot.explain import shap_values
from blindspot.faithfulness import PROMO_FEATURE, deletion_insertion, inject_promo, reference_values, whatif_uplift
from blindspot.features import CATEGORICAL, FEATURES


def _model(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    for c in CATEGORICAL:
        X[c] = pd.Categorical(rng.integers(0, 4, n))
    X[PROMO_FEATURE] = rng.integers(0, 2, n)
    # Offset keeps the target positive, so forecasts clipped at 0 are not affected.
    y = 10 + 3 * X["roll_mean_7"] + 2 * X["discount"] + 1.5 * X[PROMO_FEATURE] + rng.normal(0, 0.1, n)
    feats = FEATURES + [PROMO_FEATURE]
    b = lgb.train({"verbosity": -1, "seed": 0, "num_leaves": 15}, lgb.Dataset(X[feats], y, categorical_feature=CATEGORICAL), 100)
    return b, X, feats


def test_shap_order_beats_random():
    b, X, feats = _model()
    Xs = X.iloc[:200]
    sv, _ = shap_values(b, Xs, feats)
    r = deletion_insertion(b, Xs, sv, reference_values(X, feats), feats, n_random=3)
    assert (r.del_shap - r.del_random).mean() > 0.1
    assert (r.ins_shap - r.ins_random).mean() > 0.1


def test_whatif_recovers_additive_effect():
    b, X, feats = _model()
    assert abs(whatif_uplift(b, X.iloc[:500], feats).mean() - 1.5) < 0.2


def test_inject_promo_scales_daily_and_hourly():
    hs = [np.ones(24) for _ in range(1000)]
    p = pd.DataFrame({"sale_amount": [24.0] * 1000, "hours_sale": hs})
    q = inject_promo(p, rate=0.3, effect=0.5, seed=1)
    on = q[PROMO_FEATURE] == 1
    assert 0.25 < on.mean() < 0.35
    assert np.allclose(q.loc[on, "sale_amount"], 36.0) and np.allclose(q.loc[~on, "sale_amount"], 24.0)
    assert np.allclose([h.sum() for h in q.hours_sale], q.sale_amount)
    assert np.allclose(q.demand_no_promo, 24.0)
    assert np.allclose(p.sale_amount, 24.0)  # Input DataFrame must remain unmodified
