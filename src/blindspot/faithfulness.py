"""Faithfulness evaluation for SHAP explanations.

Implements two complementary evaluations:
1. Deletion and insertion curves: progressively replace (or restore) features
   with training reference values in descending |SHAP| order vs random order,
   scoring per-row normalized AUC.
2. Injected promotion recovery: adds an independent synthetic binary feature
   `test_promo` that scales daily and hourly demand by (1 + effect), rebuilds
   conditions A/B/C from the modified demand, and compares each model's
   estimated uplift against the known ground-truth effect.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

PROMO_FEATURE = "test_promo"


def reference_values(train: pd.DataFrame, features: list[str]) -> dict:
    """Compute neutral replacement values: training mode for categoricals, median for numerics."""
    ref = {}
    for f in features:
        s = train[f]
        ref[f] = s.mode().iloc[0] if isinstance(s.dtype, pd.CategoricalDtype) else float(s.median())
    return ref


def _replace(X: pd.DataFrame, mask: np.ndarray, ref: dict, features: list[str]) -> pd.DataFrame:
    """Return a copy of `X[features]` with entries where `mask` is True set to `ref[f]`."""
    Z = X[features].copy()
    for j, f in enumerate(features):
        m = mask[:, j]
        if m.any():
            Z.loc[m, f] = ref[f]
    return Z


def deletion_insertion(booster: lgb.Booster, X: pd.DataFrame, sv: pd.DataFrame, ref: dict, features: list[str],
                       n_random: int = 5, seed: int = 0, min_gap: float = 0.01) -> pd.DataFrame:
    """Compute per-row deletion and insertion AUCs under |SHAP| ordering vs `n_random` random orderings.

    Rows where full feature replacement changes the prediction by less than
    `min_gap` are excluded to avoid dividing by near-zero denominators.
    Mean curves across retained rows are stored in `result.attrs["curves"]`.
    """
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
        # Invert permutation `order` to get each feature's rank position per row.
        rank = np.empty_like(order)
        rank[np.arange(n)[:, None], order] = np.arange(p)[None, :]
        dele, ins = np.zeros((n, p)), np.zeros((n, p))
        for k in range(1, p + 1):
            removed = rank < k
            f_del = booster.predict(_replace(X, removed, ref, features))
            f_ins = booster.predict(_replace(X, ~removed, ref, features))
            # Normalize trajectories to move from 0 toward 1 as features are deleted or inserted.
            dele[:, k - 1] = (f_del - f_x) / np.where(keep, gap, 1)
            ins[:, k - 1] = 1 - (f_ins - f_x) / np.where(keep, gap, 1)
        out[name] = (np.clip(dele, -1, 2).mean(axis=1), np.clip(ins, -1, 2).mean(axis=1), dele, ins)

    res = pd.DataFrame({"del_shap": out["shap"][0], "ins_shap": out["shap"][1],
                        "del_random": np.mean([out[f"rand{r}"][0] for r in range(n_random)], axis=0),
                        "ins_random": np.mean([out[f"rand{r}"][1] for r in range(n_random)], axis=0)}, index=X.index)
    res.attrs["curves"] = {"del_shap": out["shap"][2][keep].mean(axis=0), "ins_shap": out["shap"][3][keep].mean(axis=0),
                           "del_random": np.mean([out[f"rand{r}"][2][keep].mean(axis=0) for r in range(n_random)], axis=0),
                           "ins_random": np.mean([out[f"rand{r}"][3][keep].mean(axis=0) for r in range(n_random)], axis=0)}
    return res[keep]


def inject_promo(panel: pd.DataFrame, rate: float = 0.2, effect: float = 0.3, seed: int = 7) -> pd.DataFrame:
    """Return a copy of `panel` with a synthetic `test_promo` flag and demand scaled by `(1 + effect)` on flagged rows.

    Scales both `sale_amount` and `hours_sale` so subsequent hourly stockout
    censoring stays consistent with daily totals, while preserving the unscaled
    baseline in `demand_no_promo`.
    """
    p = panel.copy()
    rng = np.random.default_rng(seed)
    flag = rng.random(len(p)) < rate
    p[PROMO_FEATURE] = flag.astype("int8")
    p["demand_no_promo"] = p["sale_amount"].to_numpy()
    mult = np.where(flag, 1 + effect, 1.0)
    p["sale_amount"] = p["sale_amount"].to_numpy() * mult
    p["hours_sale"] = [h * m for h, m in zip(p["hours_sale"].to_numpy(), mult)]
    return p


def train_injected(panel: pd.DataFrame, cfg: dict, rate: float = 0.2, effect: float = 0.3, seed: int = 7,
                   cond_cfg=None, conditions: str = "ABC"):
    """Inject `test_promo`, rebuild conditions and features, and train LightGBM models for `conditions`.

    Returns `(boosters, test_df, conditions_df, injected_panel)`. Because eval
    inputs are identical across conditions, a single test frame is returned.
    """
    from blindspot.conditions import ConditionConfig, build_conditions
    from blindspot.features import FEATURES, build_features
    from blindspot.model import split_rows, train_final

    pi = inject_promo(panel, rate=rate, effect=effect, seed=seed)
    cond, _ = build_conditions(pi, cond_cfg or ConditionConfig())
    feats = FEATURES + [PROMO_FEATURE]
    flag = pi[["series_id", "dt", PROMO_FEATURE, "demand_no_promo"]]
    boosters, test = {}, None
    for c in conditions:
        f = build_features(cond, pi, c).merge(flag, on=["series_id", "dt"], how="left")
        _, _, train, t = split_rows(f)
        boosters[c] = train_final(cfg["params"], cfg["n_rounds"], train, feats)
        if test is None:
            test = t
    return boosters, test, cond, pi


def whatif_uplift(booster: lgb.Booster, X: pd.DataFrame, features: list[str], feature: str = PROMO_FEATURE) -> np.ndarray:
    """Forecast difference when toggling `feature` from 0 to 1 on each row.

    Forecasts are clipped at 0, as everywhere else in the project (`model.predict`, `whatif.whatif`), so the
    uplift reported here equals the one used in Steps 9-11."""
    on, off = X[features].copy(), X[features].copy()
    on[feature], off[feature] = 1, 0
    return np.clip(booster.predict(on), 0, None) - np.clip(booster.predict(off), 0, None)
