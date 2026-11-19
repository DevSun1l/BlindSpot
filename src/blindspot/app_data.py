"""Step 12: data helpers for the Streamlit app (pure functions, no Streamlit imports, so they can be tested).

The app only READS what Steps 3-11 saved; it never retrains anything.
"""
from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
import pandas as pd

from blindspot.data import PROJECT_ROOT
from blindspot.explain import GROUP_OF, importance_share, local_stability
from blindspot.features import FEATURE_GROUPS, FEATURES
from blindspot.inventory import critical_ratio, outcomes, safety_quantiles
from blindspot.prepare import PROCESSED_DIR

OUT = PROJECT_ROOT / "outputs"
MODELS_DIR = OUT / "models"
CONDITIONS = {"A": "A: clean", "B": "B: censored", "C": "C: matched noise"}
COLORS = {"A": "#4C78A8", "B": "#E45756", "C": "#F58518"}
HISTORY_COLS = ["series_id", "dt", "split", "sales_true", "is_clean_day", "is_masked", "discount", "activity_flag"]


# ---------------------------------------------------------------- loading
def load_assets() -> dict:
    """Everything the app needs, read from disk once."""
    feats = pd.read_parquet(PROCESSED_DIR / "features_A.parquet")
    cond = pd.read_parquet(PROCESSED_DIR / "conditions.parquet",
                           columns=["series_id", "dt", "sales_B", "sales_C", "is_stockout_day", "stockout_hours"])
    hist = feats[HISTORY_COLS + ["store_id", "product_id", "sales_group"]].merge(cond, on=["series_id", "dt"], how="left")
    test = feats[feats.is_test == 1].reset_index(drop=True)
    boosters = {c: lgb.Booster(model_file=str(MODELS_DIR / f"lgb_{c}.txt")) for c in "ABC"}
    for c in "ABC":
        test[f"pred_{c}"] = np.clip(boosters[c].predict(test[FEATURES]), 0, None)
    shap_global = {c: pd.read_parquet(PROCESSED_DIR / f"shap_{c}.parquet") for c in "ABC"}
    return {"hist": hist, "test": test, "boosters": boosters, "shap_global": shap_global,
            "val_errors": pd.read_parquet(PROCESSED_DIR / "validation_errors.parquet"),
            "faithfulness": _read_csv(OUT / "step8_results.csv"),
            "seed_results": _read_csv(OUT / "step11_seed_results.csv"),
            "robustness": _read_csv(OUT / "step11_results_table.csv"),
            "noise_floor": noise_floor_from_report(OUT / "step7_shap_report.md")}


def _read_csv(path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def noise_floor_from_report(path) -> float | None:
    """Per-row SHAP agreement between A and its retrained copies (A'), from the Step 7 report."""
    if not path.exists():
        return None
    m = re.search(r"\| A vs A' \(mean of \d+\) \| ([0-9.]+) \(", path.read_text(encoding="utf-8"))
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------- series selection
def series_options(test: pd.DataFrame) -> pd.DataFrame:
    """One row per series with what makes it interesting in the test week."""
    clean = test[test.is_clean_day == 1]
    g = test.groupby("series_id", observed=True)
    out = pd.DataFrame({
        "store_id": g.store_id.first().astype(int), "product_id": g.product_id.first().astype(int),
        "sales_group": g.sales_group.first().astype(str), "promo_days": g.activity_flag.sum().astype(int),
        "clean_days": g.is_clean_day.sum().astype(int)})
    gap = clean.assign(gap=(clean.pred_A - clean.pred_B)).groupby("series_id").gap.mean()
    out["mean_gap_A_minus_B"] = gap.reindex(out.index).fillna(0)
    return out.reset_index()


def showcase_series(opts: pd.DataFrame) -> int:
    """Default example: a high-demand series with promo days and clean test days, where A and B disagree most."""
    pool = opts[(opts.promo_days > 0) & (opts.clean_days >= 3) & (opts.sales_group == "high")]
    pool = pool if len(pool) else opts
    return int(pool.sort_values("mean_gap_A_minus_B", ascending=False).series_id.iloc[0])


def history(hist: pd.DataFrame, series_id: int, train_days: int = 42) -> pd.DataFrame:
    """Last `train_days` training days plus the 7 test days for one series."""
    h = hist[hist.series_id == series_id].sort_values("dt")
    tr, ev = h[h.split == "train"].tail(train_days), h[h.split == "eval"]
    return pd.concat([tr, ev]).reset_index(drop=True)


# ---------------------------------------------------------------- explanation
def explain_row(explainers: dict, row: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """SHAP values of one row under each model. Returns (features x models frame, base values)."""
    sv, base = {}, {}
    for c, ex in explainers.items():
        sv[c] = ex.shap_values(row[FEATURES])[0]
        base[c] = float(np.ravel(ex.expected_value)[0])
    return pd.DataFrame(sv, index=FEATURES), base


def top_drivers(sv: pd.DataFrame, k: int = 8, grouped: bool = False) -> pd.DataFrame:
    """Top-k features (or the feature groups) by largest |SHAP| in any model, sorted by A's |SHAP|."""
    if grouped:
        sv = sv.groupby(GROUP_OF).sum().reindex(list(FEATURE_GROUPS))
    keep = sv.abs().max(axis=1).nlargest(min(k, len(sv))).index
    return sv.loc[keep].reindex(sv.loc[keep, "A"].abs().sort_values().index)


def row_stability(sv: pd.DataFrame) -> pd.DataFrame:
    """Agreement of this row's explanation between A and B, and between A and C."""
    rows = []
    for c in "BC":
        s = local_stability(sv[["A"]].T.reset_index(drop=True), sv[[c]].T.reset_index(drop=True))
        rows.append({"comparison": f"A vs {c}", "spearman": float(s.spearman.iloc[0]),
                     "top3_overlap": float(s.top3_overlap.iloc[0])})
    return pd.DataFrame(rows)


def global_shares(shap_global: dict, grouped: bool = False) -> pd.DataFrame:
    """Share of total mean |SHAP| per feature (or group), per model, from Step 7's explained rows."""
    out = {}
    for c, df in shap_global.items():
        sv = df[FEATURES]
        if grouped:
            sv = sv.T.groupby(GROUP_OF).sum().T
        out[c] = importance_share(sv)
    return pd.DataFrame(out).sort_values("A", ascending=False)


def global_stability(shap_global: dict) -> pd.DataFrame:
    rows = []
    for c in "BC":
        s = local_stability(shap_global["A"][FEATURES], shap_global[c][FEATURES])
        rows.append({"comparison": f"A vs {c}", "mean per-row Spearman": s.spearman.mean(),
                     "mean top-3 overlap": s.top3_overlap.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ordering
def safety_for(val_errors: pd.DataFrame, cu: float, co: float) -> dict:
    """Safety allowance per model and sales group for the chosen costs (same rule as Step 10)."""
    cr = critical_ratio(cu, co)
    return {c: safety_quantiles(v.err_own.to_numpy(), v.group.to_numpy(), cr)
            for c, v in val_errors.groupby("condition")}


def order_table(forecasts: dict, group: str, safety: dict, cu: float, co: float,
                true_demand: float | None) -> pd.DataFrame:
    """Recommended order per model; outcome columns only when the true demand is known."""
    rows = []
    for c, f in forecasts.items():
        s = safety[c][group]
        q = max(0.0, f + s)
        r = {"model": CONDITIONS[c], "forecast": f, "safety allowance": s, "recommended order": q}
        if true_demand is not None:
            o = outcomes(np.array([q]), np.array([true_demand]), cu, co).iloc[0]
            r.update({"true demand": true_demand, "lost sales": o.short, "wasted": o.waste, "cost": o.cost})
        rows.append(r)
    return pd.DataFrame(rows)


def cost_curve(orders: dict, demand_max: float, cu: float, co: float, n: int = 101) -> pd.DataFrame:
    """Cost of each model's order if demand turned out to be d, for d in [0, demand_max]."""
    d = np.linspace(0, demand_max, n)
    return pd.DataFrame({"demand": d, **{c: outcomes(np.full(n, q), d, cu, co).cost.to_numpy()
                                         for c, q in orders.items()}})
