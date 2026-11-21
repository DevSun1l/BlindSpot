"""Alternative approaches to training on stockout-censored sales (Step 17).

All approaches start from the censored data (condition B). Like a real retailer with hourly stock status, approaches
1-3 know WHICH training days stocked out and AT WHAT HOUR (`is_masked`, `sim_stockout_hour`), but never the true
demand on those days.

  naive           train on recorded sales as if they were demand (= condition B)
  drop            exclude stockout days as training targets (they still feed lag features as recorded)
  impute          replace each stockout day's sales with an estimate of full-day demand (unconstraining)
  censored_loss   keep stockout days but treat their sales as a lower bound: under-forecasting them is penalised,
                  over-forecasting is not (Tobit-style one-sided squared loss)

Approaches 1-3 compute the ordering safety allowance from validation days without a stockout, since those are the
only days whose sales equal demand. Real stockout days are handled identically in every approach and in A.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from blindspot.conditions import ConditionConfig, build_conditions
from blindspot.experiment import fit_bundle, score_models
from blindspot.faithfulness import PROMO_FEATURE, inject_promo
from blindspot.features import CATEGORICAL, FEATURES, build_features
from blindspot.model import base_params, train_final

APPROACHES = ["naive", "drop", "impute", "censored_loss"]
LABELS = {"A": "clean (reference)", "naive": "1. naive (= B)", "drop": "2. drop stockout days",
          "impute": "3. impute full-day demand", "censored_loss": "4. censored loss"}
# Below this share of the day's usual sales, scaling up the sales before the stockout is too unstable.
MIN_SHARE = 0.25
# Pseudo-units of the store-wide hourly pattern mixed into each series' own pattern.
PROFILE_PRIOR = 20.0


# ---------------------------------------------------------------- imputation
def hourly_share(cond: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Per series, the usual share of a day's sales made before each hour (columns 0..24).

    Learned only from training days with no stockout (neither real nor simulated), whose hourly sales are complete.
    Each series' pattern is shrunk toward the pooled pattern by `PROFILE_PRIOR` units, so sparse series stay stable.
    """
    ok = cond[(cond.split == "train") & (cond.is_clean_day == 1) & (cond.is_masked == 0)][["series_id", "dt"]]
    h = ok.merge(panel[["series_id", "dt", "hours_sale"]], on=["series_id", "dt"], how="left")
    hs = pd.DataFrame(np.stack(h.hours_sale.to_numpy()), index=h.series_id.to_numpy()).groupby(level=0).sum()
    pooled = hs.sum() / hs.sum().sum()
    tot = hs.sum(axis=1).to_numpy()[:, None]
    mix = (hs.to_numpy() + PROFILE_PRIOR * pooled.to_numpy()[None, :]) / (tot + PROFILE_PRIOR)
    cum = np.concatenate([np.zeros((len(hs), 1)), np.cumsum(mix, axis=1)], axis=1)   # share sold before hour h
    return pd.DataFrame(cum, index=hs.index)


def impute_demand(cond: pd.DataFrame, panel: pd.DataFrame) -> np.ndarray:
    """Estimated full-day demand for every row: censored sales on non-stockout rows are returned unchanged.

    On a stockout day starting at hour h with s units sold before h, and with `q` the series' usual share of daily
    sales made before h:
      q >= MIN_SHARE: demand = s / q            (scale the morning up to a full day)
      q <  MIN_SHARE: demand = s + (1 - q) * m  (too early to scale: fill the rest with the series' usual day, m)
    where m is the series' mean sales on training days without a stockout.
    """
    out = cond.sales_B.to_numpy(dtype=float).copy()
    rows = np.flatnonzero(cond.is_masked.to_numpy() == 1)
    if not len(rows):
        return out
    share = hourly_share(cond, panel)
    ok = (cond.split == "train") & (cond.is_clean_day == 1) & (cond.is_masked == 0)
    m = cond[ok].groupby("series_id").sales_B.mean()
    sid = cond.series_id.to_numpy()[rows]
    hour = cond.sim_stockout_hour.to_numpy()[rows].astype(int)
    pooled = share.mean().to_numpy()                      # fallback for a series with no complete day
    q = share.reindex(sid).to_numpy()[np.arange(len(rows)), hour]
    q = np.where(np.isnan(q), pooled[hour], q)
    s = out[rows]
    mean_day = m.reindex(sid).fillna(0).to_numpy()
    out[rows] = np.where(q >= MIN_SHARE, s / np.maximum(q, MIN_SHARE), s + (1 - q) * mean_day)
    return np.maximum(out, cond.sales_B.to_numpy())   # demand is never below what was sold


# ---------------------------------------------------------------- censored loss
class ShiftedBooster:
    """A LightGBM booster trained on targets minus `shift`; `.predict` adds it back.

    A custom objective in LightGBM starts from 0 instead of the target mean, so targets are centred first. With no
    censored rows and no feature/row subsampling, the custom loss then reproduces the built-in regression objective
    (tested). With subsampling, LightGBM draws a different random stream for custom objectives, which acts like a
    different random seed; the noise floor in Step 7 measures how much that alone moves results.
    """

    def __init__(self, booster: lgb.Booster, shift: float):
        self.booster, self.shift = booster, float(shift)

    def predict(self, X, **kw):
        return self.booster.predict(X, **kw) + self.shift


def censored_objective(censored: np.ndarray):
    """One-sided squared loss: on censored rows the label is a lower bound, so forecasts above it cost nothing."""
    cen = censored.astype(bool)

    def fobj(preds, data):
        grad = preds - data.get_label()
        grad[cen & (grad > 0)] = 0.0
        return grad, np.ones_like(grad)
    return fobj


def train_censored(params: dict, n_rounds: int, train: pd.DataFrame, features: list[str] | None = None,
                   censored_col: str = "censored") -> ShiftedBooster:
    """Train LightGBM with `censored_objective`, treating rows where `censored_col` == 1 as lower bounds."""
    feats = features or FEATURES
    shift = float(train.target.mean())
    p = base_params({k: v for k, v in params.items() if k not in ("n_estimators", "objective")})
    p["objective"] = censored_objective(train[censored_col].to_numpy())
    ds = lgb.Dataset(train[feats], label=train.target - shift, categorical_feature=CATEGORICAL, free_raw_data=False)
    return ShiftedBooster(lgb.train(p, ds, num_boost_round=n_rounds), shift)


# ---------------------------------------------------------------- approach features and models
def approach_features(cond: pd.DataFrame, panel: pd.DataFrame, approach: str) -> pd.DataFrame:
    """Feature table for one approach, built from the censored sales (condition B)."""
    if approach in ("naive", "drop", "censored_loss"):
        f = build_features(cond, panel, "B")
    elif approach == "impute":
        f = build_features(cond.assign(sales_Bimp=impute_demand(cond, panel)), panel, "Bimp")
    else:
        raise ValueError(f"unknown approach {approach!r}")
    stock = (f.is_masked == 1) & (f.split == "train")
    if approach == "drop":
        f["is_train_target"] = (f.is_train_target.astype(bool) & ~stock).astype("int8")
        f["is_valid"] = (f.is_valid.astype(bool) & ~stock).astype("int8")
    f["censored"] = (stock & (approach == "censored_loss")).astype("int8")
    f["condition"] = approach
    return f


def approach_bundle(f: pd.DataFrame, cfg: dict, approach: str, features: list[str] | None = None) -> dict:
    """Train the approach's models (see `experiment.fit_bundle`)."""
    trainer = train_censored if approach == "censored_loss" else train_final
    knows_stockouts = None if approach == "naive" else (lambda v: v.is_masked == 0)
    return fit_bundle(f, cfg, trainer, features, resid_rows=knows_stockouts)


def run_approaches(panel: pd.DataFrame, mask_seed: int, cfg: dict, a_cache: dict | None = None,
                   approaches: list[str] = APPROACHES, explain_rows: int = 2000, effect: float = 0.3,
                   rate: float = 0.2) -> pd.DataFrame:
    """All headline metrics for the clean reference A and each approach, at one masking seed.

    Uses the same masking (30% demand-linked), injected promotion and scoring as Step 11, so `naive` reproduces B.
    `a_cache` is shared with `experiment.run_seed` (keys "A" and "A_inj").
    """
    from blindspot.experiment import _models
    a_cache = {} if a_cache is None else a_cache
    cc = ConditionConfig(seed=mask_seed)
    cond, _ = build_conditions(panel, cc)
    if "A" not in a_cache:
        a_cache["A"] = _models(cond, panel, "A", cfg)
    m = {"A": a_cache["A"]}
    for ap in approaches:
        m[ap] = approach_bundle(approach_features(cond, panel, ap), cfg, ap)

    # Injected promotion: same injection and masking as Step 8/11, then each approach on the injected data.
    pi = inject_promo(panel, rate=rate, effect=effect, seed=7)
    cond_i, _ = build_conditions(pi, cc)
    flag = pi[["series_id", "dt", PROMO_FEATURE, "demand_no_promo"]]
    feats = FEATURES + [PROMO_FEATURE]
    inj, t_inj = {}, None
    for ap in ["A"] + list(approaches):
        if ap == "A" and "A_inj" in a_cache:
            continue
        base = build_features(cond_i, pi, "A") if ap == "A" else approach_features(cond_i, pi, ap)
        f = base.merge(flag, on=["series_id", "dt"], how="left")
        tr = f[f.is_train_target == 1]
        trainer = train_censored if ap == "censored_loss" else train_final
        inj[ap] = trainer(cfg["params"], cfg["n_rounds"], tr, feats)
        if t_inj is None:
            t_inj = f[f.is_test == 1]
        if ap == "A":
            a_cache["A_inj"] = inj["A"]
    inj["A"] = a_cache["A_inj"]
    return score_models(m, inj, t_inj, mask_seed, explain_rows, effect)


def imputation_quality(cond: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """How close imputed demand is to the true demand on simulated stockout days (known only in this study)."""
    imp = impute_demand(cond, panel)
    r = cond.is_masked.to_numpy() == 1
    t, b, i = cond.sales_true.to_numpy()[r], cond.sales_B.to_numpy()[r], imp[r]
    return {"days": int(r.sum()), "true_total": float(t.sum()),
            "censored_shortfall": float((b - t).sum() / t.sum()), "imputed_shortfall": float((i - t).sum() / t.sum()),
            "censored_wape": float(np.abs(b - t).sum() / t.sum()), "imputed_wape": float(np.abs(i - t).sum() / t.sum())}
