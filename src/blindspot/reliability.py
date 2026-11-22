"""Reliability warning for explanations (Step 18): flag items whose SHAP explanation is likely distorted.

Task (binary classification, one row per item = store x product series):
  label 1 "unreliable"  the deployed model's explanation agrees with the clean model A less than retraining noise
                        allows: the item's mean per-row SHAP Spearman vs A is below the `FLOOR_Q` quantile of the
                        same statistic for A retrained with other seeds (A'). Known only in this study.
  label 0 "reliable"    otherwise.

Warning features use only what a retailer has without the true demand: stockout records, promotion history,
recorded sales, and two models it can train itself (naive and censored loss) - never model A.

Deployed models scored: `naive` (trained on recorded sales) and `censored_loss` (Step 17's recommendation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from blindspot.conditions import ConditionConfig, build_conditions
from blindspot.explain import local_stability, shap_values
from blindspot.features import build_features
from blindspot.mitigation import approach_features, train_censored
from blindspot.model import predict, split_rows, train_final

REF_SEEDS = [43, 44]          # LightGBM seeds for the retrained clean models A' (noise floor), as in Step 7
FLOOR_Q = 0.05                # unreliable = worse than 95% of items under retraining noise alone
DEPLOYED = ["naive", "censored_loss"]
FEATURES = ["stockout_share", "hours_out_share", "promo_share", "discount_mean", "sales_mean", "sales_cv",
            "low_sales", "model_disagreement", "forecast_gap"]
FEATURE_LABELS = {
    "stockout_share": "share of training days with a stockout",
    "hours_out_share": "share of business hours out of stock",
    "promo_share": "share of training days with a promotion",
    "discount_mean": "mean discount level",
    "sales_mean": "mean recorded daily sales",
    "sales_cv": "variability of recorded sales",
    "low_sales": "low-sales item",
    "model_disagreement": "naive vs censored-loss explanation disagreement",
    "forecast_gap": "naive vs censored-loss forecast gap",
}
BUSINESS_HOURS = 16


# ---------------------------------------------------------------- data per masking seed
def _train(f: pd.DataFrame, cfg: dict, censored: bool = False, params: dict | None = None):
    _, _, train, test = split_rows(f)
    p = params or cfg["params"]
    model = (train_censored if censored else train_final)(p, cfg["n_rounds"], train)
    return model, test


def per_series_agreement(sv_ref: pd.DataFrame, sv_m: pd.DataFrame, series: np.ndarray) -> pd.Series:
    """Mean per-row SHAP rank agreement (Spearman of |SHAP|) per series."""
    rho = local_stability(sv_ref, sv_m).spearman.to_numpy()
    return pd.Series(rho).groupby(series).mean()


def item_table(panel: pd.DataFrame, mask_seed: int, cfg: dict, cache: dict | None = None) -> pd.DataFrame:
    """One row per series: warning features plus the explanation agreement of each deployed model and of A'."""
    cache = {} if cache is None else cache
    cond, _ = build_conditions(panel, ConditionConfig(seed=mask_seed))
    if "A" not in cache:
        fa = build_features(cond, panel, "A")
        cache["A"], test = _train(fa, cfg)
        cache["test"] = test[test.is_clean_day == 1].reset_index(drop=True)
        X = cache["test"]
        cache["sv_A"] = shap_values(cache["A"], X)[0]
        for s in REF_SEEDS:
            ref, _ = _train(fa, cfg, params={**cfg["params"], "seed": s})
            cache[f"sv_A'{s}"] = shap_values(ref, X)[0]
    X = cache["test"]
    sid = X.series_id.to_numpy()

    models = {ap: _train(approach_features(cond, panel, ap), cfg, censored=ap == "censored_loss")[0] for ap in DEPLOYED}
    sv = {ap: shap_values(m, X)[0] for ap, m in models.items()}
    pred = {ap: predict(m, X) for ap, m in models.items()}

    out = pd.DataFrame(index=pd.Index(np.unique(sid), name="series_id"))
    for ap in DEPLOYED:
        out[f"agree_{ap}"] = per_series_agreement(cache["sv_A"], sv[ap], sid)
    for s in REF_SEEDS:
        out[f"agree_ref{s}"] = per_series_agreement(cache["sv_A"], cache[f"sv_A'{s}"], sid)
    out["model_disagreement"] = 1 - per_series_agreement(sv["naive"], sv["censored_loss"], sid)
    gap = pd.DataFrame({"g": np.abs(pred["naive"] - pred["censored_loss"]), "m": pred["censored_loss"], "s": sid})
    gap = gap.groupby("s")[["g", "m"]].sum()
    out["forecast_gap"] = gap.g / gap.m.clip(lower=1e-6)

    # Training-history features, from information a retailer has (stock records, promotions, recorded sales).
    tr = cond[cond.split == "train"].merge(panel[["series_id", "dt", "activity_flag", "discount", "sales_group"]],
                                           on=["series_id", "dt"], how="left")
    stock = (tr.is_stockout_day == 1) | (tr.is_masked == 1)
    hours = np.where(tr.is_masked == 1, 22 - tr.sim_stockout_hour, tr.stockout_hours).clip(0, BUSINESS_HOURS)
    g = tr.assign(stock=stock.astype(float), hours=hours / BUSINESS_HOURS, low=(tr.sales_group == "low").astype(float)
                  ).groupby("series_id")
    out["stockout_share"] = g.stock.mean()
    out["hours_out_share"] = g.hours.mean()
    out["promo_share"] = g.activity_flag.mean()
    out["discount_mean"] = g.discount.mean()
    out["sales_mean"] = g.sales_B.mean()
    out["sales_cv"] = g.sales_B.std() / g.sales_B.mean().clip(lower=1e-6)
    out["low_sales"] = g.low.first()
    out["store_id"] = g.store_id.first()
    out["n_test_rows"] = pd.Series(sid).value_counts()
    out["seed"] = mask_seed
    return out.reset_index()


def floor_threshold(items: pd.DataFrame, q: float = FLOOR_Q) -> float:
    """Agreement below which an item is worse than retraining noise (pooled A vs A' items, quantile `q`)."""
    return float(np.quantile(np.concatenate([items[f"agree_ref{s}"].to_numpy() for s in REF_SEEDS]), q))


def add_labels(items: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add `unreliable_<model>` labels (1 = explanation beyond the noise floor)."""
    items = items.copy()
    for ap in DEPLOYED:
        items[f"unreliable_{ap}"] = (items[f"agree_{ap}"] < threshold).astype(int)
    return items


# ---------------------------------------------------------------- warning methods
def make_methods(seed: int = 0) -> dict:
    """Candidate warnings. Single-feature rules need no training (their score is the feature itself)."""
    return {
        "rule: stockout share": "stockout_share",
        "rule: model disagreement": "model_disagreement",
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "gradient boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                            class_weight="balanced", random_state=seed),
    }


def fit_score(method, train: pd.DataFrame, test: pd.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (train scores, test scores) for one warning method."""
    if isinstance(method, str):
        return train[method].to_numpy(), test[method].to_numpy()
    method.fit(train[FEATURES], train[label])
    return method.predict_proba(train[FEATURES])[:, 1], method.predict_proba(test[FEATURES])[:, 1]


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Score threshold that maximises F1 (chosen on training data only)."""
    p, r, t = precision_recall_curve(y, score)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    return float(t[np.argmax(f1)])


def classification_report(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """ROC AUC, PR AUC and threshold metrics (accuracy, precision, recall, specificity, F1, confusion matrix)."""
    yhat = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    n = len(y)
    return {"n": n, "positives": int(y.sum()), "prevalence": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score)),
            "threshold": threshold, "accuracy": (tp + tn) / n,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0, "f1": float(f1_score(y, yhat, zero_division=0)),
            "majority_accuracy": max(y.mean(), 1 - y.mean()), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, clusters: np.ndarray, n_boot: int = 500,
                     seed: int = 0) -> tuple[float, float]:
    """95% CI for ROC AUC, resampling clusters (stores) with replacement."""
    rng = np.random.default_rng(seed)
    uc, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uc))]
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uc), len(uc))])
        if 0 < y[idx].sum() < len(idx):
            aucs.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def store_split(items: pd.DataFrame, frac: float = 0.5, seed: int = 0) -> np.ndarray:
    """Boolean mask: True for items whose store is in the training half (stores never straddle the split)."""
    stores = np.sort(items.store_id.unique())
    rng = np.random.default_rng(seed)
    train_stores = set(rng.choice(stores, size=int(round(len(stores) * frac)), replace=False))
    return items.store_id.isin(train_stores).to_numpy()


def roc_points(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(y, score)
    return fpr, tpr
