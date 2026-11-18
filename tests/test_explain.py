"""Step 7 tests: SHAP additivity, grouping and stability measures on small synthetic data."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from blindspot.explain import (bootstrap_mean_ci, global_importance, global_stability, grouped, local_stability,
                               shap_values, topk_overlap)
from blindspot.features import CATEGORICAL, FEATURE_GROUPS, FEATURES


def _toy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    for c in CATEGORICAL:
        X[c] = pd.Categorical(rng.integers(0, 4, n))
    y = 3 * X["roll_mean_7"] + 1 * X["discount"] + rng.normal(0, 0.1, n)
    b = lgb.train({"verbosity": -1, "seed": 0, "num_leaves": 15}, lgb.Dataset(X, y, categorical_feature=CATEGORICAL), 50)
    return b, X


def test_shap_additivity_and_ranking():
    b, X = _toy()
    sv, base = shap_values(b, X.iloc[:200])
    assert np.allclose(base + sv.sum(axis=1), b.predict(X.iloc[:200]), atol=1e-6)
    imp = global_importance(sv)
    assert list(imp.index[:2]) == ["roll_mean_7", "discount"]


def test_grouped_preserves_row_sums():
    b, X = _toy()
    sv, _ = shap_values(b, X.iloc[:100])
    g = grouped(sv)
    assert list(g.columns) == list(FEATURE_GROUPS)
    assert np.allclose(g.sum(axis=1), sv.sum(axis=1))


def test_stability_measures():
    rng = np.random.default_rng(1)
    a = pd.DataFrame(rng.normal(size=(50, 6)), columns=list("abcdef"))
    same = local_stability(a, a)
    assert np.allclose(same.spearman, 1) and np.allclose(same.top3_overlap, 1)
    imp = pd.Series([5, 4, 3, 2, 1, 0.5], index=list("abcdef"))
    assert global_stability(imp, imp)["spearman"] == 1.0
    rev = pd.Series(imp.values[::-1], index=imp.index).sort_values(ascending=False)
    assert global_stability(imp, rev)["spearman"] == -1.0
    assert topk_overlap(imp, rev, k=2) == 0


def test_importance_share_is_scale_free():
    from blindspot.explain import importance_share
    sv = pd.DataFrame({"a": [1.0, -2.0], "b": [0.5, 0.5]})
    s1, s2 = importance_share(sv), importance_share(sv * 0.5)
    assert np.isclose(s1.sum(), 1) and np.allclose(s1, s2)


def test_bootstrap_mean_ci():
    v = np.random.default_rng(0).normal(0.8, 0.1, 1000)
    lo, hi = bootstrap_mean_ci(v, np.arange(1000) % 50)
    assert lo < v.mean() < hi
