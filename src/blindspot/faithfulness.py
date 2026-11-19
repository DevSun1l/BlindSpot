"""Step 8: faithfulness tests for SHAP explanations.

1. Deletion / insertion curves
   Deletion: replace features with a neutral reference value one at a time, in order of |SHAP|
   (largest first). If SHAP is faithful, the forecast should move away from its original value
   faster than when features are removed in random order.
   Insertion: start from the all-reference input and restore features in |SHAP| order; the forecast
   should return to its original value faster than in random order.
   Reference values: median (numeric) / most frequent value (categorical) of condition A's training rows.
   Score per row: area under the normalised curve (mean over k = 1..n_features); higher = more faithful.

2. Injected promotion recovery (ground truth known)
   A synthetic binary feature `test_promo` is switched on for a random share of series-days, independent
   of everything else, and the TRUE demand (daily and hourly) is multiplied by (1 + effect) on those days.
   Conditions A/B/C are then rebuilt from this modified truth, so censoring treats promo days like any
   other high-demand day. Each model's estimate of the promo effect (SHAP and what-if) is compared with
   the known effect.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

PROMO_FEATURE = "test_promo"


def reference_values(train: pd.DataFrame, features: list[str]) -> dict:
    ref = {}
    for f in features:
        s = train[f]
        ref[f] = s.mode().iloc[0] if isinstance(s.dtype, pd.CategoricalDtype) else float(s.median())
    return ref


def _replace(X: pd.DataFrame, mask: np.ndarray, ref: dict, features: list[str]) -> pd.DataFrame:
    Z = X[features].copy()
    for j, f in enumerate(features):
        m = mask[:, j]
        if m.any():
            Z.loc[m, f] = ref[f]
    return Z


def deletion_insertion(booster: lgb.Booster, X: pd.DataFrame, sv: pd.DataFrame, ref: dict, features: list[str],
                       n_random: int = 5, seed: int = 0, min_gap: float = 0.01) -> pd.DataFrame:
    """Returns per-row AUC scores: del_shap, del_random, ins_shap, ins_random (higher = more faithful).
    Rows whose forecast barely changes when every feature is replaced (|gap| < min_gap) are dropped."""
    n, p = len(X), len(features)
    f_x = booster.predict(X[features])
    f_ref = booster.predict(_replace(X, np.ones((n, p), bool), ref, features))
    gap = f_ref - f_x
    keep = np.abs(gap) >= min_gap

    rng = np.random.default_rng(seed)
    orders = {"shap": np.argsort(-sv[features].abs().to_numpy(), axis=1)}
    for r in range(n_random):
        orders[f"rand{r}"] = np.argsort(rng.random((n, p)), axis=1)

    out = {}
    for name, order in orders.items():
        rank = np.empty_like(order)
        rank[np.arange(n)[:, None], order] = np.arange(p)[None, :]
        dele, ins = np.zeros((n, p)), np.zeros((n, p))
        for k in range(1, p + 1):
            removed = rank < k                     # top-k features (by this order) replaced
            f_del = booster.predict(_replace(X, removed, ref, features))
            f_ins = booster.predict(_replace(X, ~removed, ref, features))   # only top-k kept
            dele[:, k - 1] = (f_del - f_x) / np.where(keep, gap, 1)        # 0 -> 1 as features are removed
            ins[:, k - 1] = 1 - (f_ins - f_x) / np.where(keep, gap, 1)     # 0 -> 1 as features are restored
        out[name] = (np.clip(dele, -1, 2).mean(axis=1), np.clip(ins, -1, 2).mean(axis=1), dele, ins)

    res = pd.DataFrame({"del_shap": out["shap"][0], "ins_shap": out["shap"][1],
                        "del_random": np.mean([out[f"rand{r}"][0] for r in range(n_random)], axis=0),
                        "ins_random": np.mean([out[f"rand{r}"][1] for r in range(n_random)], axis=0)}, index=X.index)
    res.attrs["curves"] = {"del_shap": out["shap"][2][keep].mean(axis=0), "ins_shap": out["shap"][3][keep].mean(axis=0),
                           "del_random": np.mean([out[f"rand{r}"][2][keep].mean(axis=0) for r in range(n_random)], axis=0),
                           "ins_random": np.mean([out[f"rand{r}"][3][keep].mean(axis=0) for r in range(n_random)], axis=0)}
    return res[keep]


def inject_promo(panel: pd.DataFrame, rate: float = 0.2, effect: float = 0.3, seed: int = 7) -> pd.DataFrame:
    """Copy of the panel with a random `test_promo` flag and true demand scaled by (1 + effect) on flagged days.
    The original (unscaled) demand is kept as `demand_no_promo` so the true uplift per row is known."""
    p = panel.copy()
    rng = np.random.default_rng(seed)
    flag = rng.random(len(p)) < rate
    p[PROMO_FEATURE] = flag.astype("int8")
    p["demand_no_promo"] = p["sale_amount"].to_numpy()
    mult = np.where(flag, 1 + effect, 1.0)
    p["sale_amount"] = p["sale_amount"].to_numpy() * mult
    p["hours_sale"] = [h * m for h, m in zip(p["hours_sale"].to_numpy(), mult)]
    return p


def train_injected(panel: pd.DataFrame, cfg: dict, rate: float = 0.2, effect: float = 0.3, seed: int = 7):
    """Inject the synthetic promo, rebuild conditions A/B/C and their features from the modified truth, and train
    each model with the frozen config plus `test_promo`.
    Returns (boosters {c: Booster}, test rows of condition A with test_promo and demand_no_promo, condition table,
    injected panel). Test inputs are identical across conditions, so condition A's test rows serve all three."""
    from blindspot.conditions import ConditionConfig, build_conditions
    from blindspot.features import FEATURES, build_features
    from blindspot.model import split_rows, train_final

    pi = inject_promo(panel, rate=rate, effect=effect, seed=seed)
    cond, _ = build_conditions(pi, ConditionConfig())
    feats = FEATURES + [PROMO_FEATURE]
    flag = pi[["series_id", "dt", PROMO_FEATURE, "demand_no_promo"]]
    boosters, test = {}, None
    for c in "ABC":
        f = build_features(cond, pi, c).merge(flag, on=["series_id", "dt"], how="left")
        _, _, train, t = split_rows(f)
        boosters[c] = train_final(cfg["params"], cfg["n_rounds"], train, feats)
        if c == "A":
            test = t
    return boosters, test, cond, pi


def whatif_uplift(booster: lgb.Booster, X: pd.DataFrame, features: list[str], feature: str = PROMO_FEATURE) -> np.ndarray:
    """Model's what-if effect of switching `feature` from 0 to 1 on each row (no clipping)."""
    on, off = X[features].copy(), X[features].copy()
    on[feature], off[feature] = 1, 0
    return booster.predict(on) - booster.predict(off)
