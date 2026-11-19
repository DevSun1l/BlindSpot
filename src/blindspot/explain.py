"""SHAP feature attributions and cross-condition explanation stability metrics.

Uses exact `shap.TreeExplainer` with `feature_perturbation="tree_path_dependent"`,
which supports LightGBM categorical splits without an external background dataset
and satisfies exact additivity (expected_value + sum(SHAP) == raw prediction).
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
    """Draw a reproducible sample of clean test rows, stratified proportionally by sales_group."""
    clean = test[test.is_clean_day == 1]
    frac = min(1.0, n / len(clean))
    return (clean.groupby("sales_group", observed=True, group_keys=False)
            .sample(frac=frac, random_state=seed).sort_values(["series_id", "dt"]).reset_index(drop=True))


def shap_values(booster: lgb.Booster, X: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, float]:
    """Compute exact TreeSHAP values and scalar base value for `X`."""
    cols = features or FEATURES
    ex = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
    sv = ex.shap_values(X[cols])
    base = float(np.ravel(ex.expected_value)[0])
    return pd.DataFrame(sv, columns=cols, index=X.index), base


def grouped(sv: pd.DataFrame) -> pd.DataFrame:
    """Sum feature SHAP values into semantic feature groups (preserving row additivity)."""
    return sv.T.groupby(GROUP_OF).sum().T[list(FEATURE_GROUPS)]


def global_importance(sv: pd.DataFrame) -> pd.Series:
    """Return mean |SHAP| per feature, sorted descending."""
    return sv.abs().mean().sort_values(ascending=False)


def importance_share(sv: pd.DataFrame) -> pd.Series:
    """Return scale-free importance: each feature's mean |SHAP| divided by total mean |SHAP|.

    Normalizing by total |SHAP| allows comparing models whose predictions sit on
    different overall scales (e.g. censored vs clean).
    """
    imp = sv.abs().mean()
    return (imp / imp.sum()).sort_values(ascending=False)


def topk_overlap(a: pd.Series, b: pd.Series, k: int = TOP_K) -> float:
    """Fraction of shared features in the top-k entries of two ranked importance series."""
    return len(set(a.index[:k]) & set(b.index[:k])) / k


def global_stability(imp_a: pd.Series, imp_b: pd.Series, k: int = TOP_K) -> dict:
    """Rank-agreement metrics (Spearman, Kendall, top-k overlap) between two global importance profiles."""
    b = imp_b.reindex(imp_a.index)
    return {"spearman": float(spearmanr(imp_a.values, b.values).statistic),
            "kendall": float(kendalltau(imp_a.values, b.values).statistic),
            f"top{k}_overlap": topk_overlap(imp_a, imp_b, k)}


def local_stability(sv_a: pd.DataFrame, sv_b: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Compute per-row agreement between two SHAP matrices (Spearman of |SHAP| and top-k overlap)."""
    A, B = sv_a.abs().to_numpy(), sv_b.abs().to_numpy()
    # Vectorized row-wise rank transform (assumes no within-row ties).
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
    """Return a 95% cluster-bootstrap CI for the sample mean, resampling `clusters` with replacement."""
    df = pd.DataFrame({"v": values, "c": clusters}).groupby("c").v.agg(["sum", "size"])
    s, n = df["sum"].to_numpy(), df["size"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(df), size=(n_boot, len(df)))
    m = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
