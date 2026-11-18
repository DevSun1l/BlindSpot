"""Step 7: SHAP explanations and their stability across conditions.

- Explainer: shap.TreeExplainer (exact for tree models), feature_perturbation="tree_path_dependent".
  This uses each model's own trees, so no separate background sample is needed and it handles
  LightGBM categorical splits. Additivity holds: base value + sum(SHAP) = model prediction.
- The SAME fixed set of clean test rows is explained for every model.
- Grouped SHAP: by additivity, a group's contribution on a row = sum of its features' SHAP values.
- SHAP describes what the model uses, not what causes sales.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, spearmanr

from blindspot.features import FEATURE_GROUPS, FEATURES

GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
TOP_K = 5


def explain_set(test: pd.DataFrame, n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Fixed sample of clean test rows, stratified by sales group in proportion."""
    clean = test[test.is_clean_day == 1]
    frac = min(1.0, n / len(clean))
    return (clean.groupby("sales_group", observed=True, group_keys=False)
            .sample(frac=frac, random_state=seed).sort_values(["series_id", "dt"]).reset_index(drop=True))


def shap_values(booster: lgb.Booster, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    ex = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
    sv = ex.shap_values(X[FEATURES])
    base = float(np.ravel(ex.expected_value)[0])
    return pd.DataFrame(sv, columns=FEATURES, index=X.index), base


def grouped(sv: pd.DataFrame) -> pd.DataFrame:
    return sv.T.groupby(GROUP_OF).sum().T[list(FEATURE_GROUPS)]


def global_importance(sv: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per column, sorted descending."""
    return sv.abs().mean().sort_values(ascending=False)


def importance_share(sv: pd.DataFrame) -> pd.Series:
    """Scale-free importance: each feature's mean |SHAP| as a share of the total mean |SHAP|.
    Needed because a model that forecasts lower overall (e.g. B) has smaller SHAP values everywhere."""
    imp = sv.abs().mean()
    return (imp / imp.sum()).sort_values(ascending=False)


def topk_overlap(a: pd.Series, b: pd.Series, k: int = TOP_K) -> float:
    return len(set(a.index[:k]) & set(b.index[:k])) / k


def global_stability(imp_a: pd.Series, imp_b: pd.Series, k: int = TOP_K) -> dict:
    b = imp_b.reindex(imp_a.index)
    return {"spearman": float(spearmanr(imp_a.values, b.values).statistic),
            "kendall": float(kendalltau(imp_a.values, b.values).statistic),
            f"top{k}_overlap": topk_overlap(imp_a, imp_b, k)}


def local_stability(sv_a: pd.DataFrame, sv_b: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Per-row agreement between two models' explanations of the same row:
    Spearman rank correlation of |SHAP| across features, and top-k overlap of the largest |SHAP| features."""
    A, B = sv_a.abs().to_numpy(), sv_b.abs().to_numpy()
    ra = A.argsort(axis=1).argsort(axis=1).astype(float)
    rb = B.argsort(axis=1).argsort(axis=1).astype(float)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    rho = (ra * rb).sum(axis=1) / np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    ta = np.argsort(-A, axis=1)[:, :k]
    tb = np.argsort(-B, axis=1)[:, :k]
    overlap = np.array([len(set(x) & set(y)) / k for x, y in zip(ta, tb)])
    return pd.DataFrame({"spearman": rho, f"top{k}_overlap": overlap}, index=sv_a.index)


def bootstrap_mean_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """95% CI of a mean, resampling clusters (stores) with replacement."""
    df = pd.DataFrame({"v": values, "c": clusters}).groupby("c").v.agg(["sum", "size"])
    s, n = df["sum"].to_numpy(), df["size"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(df), size=(n_boot, len(df)))
    m = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
