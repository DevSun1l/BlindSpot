"""In-memory multi-seed experiment runner for conditions A, B, and C.

Evaluates forecasting accuracy, SHAP stability, what-if uplift, everyday
newsvendor costs, and injected-promotion recovery for a given masking seed
without overwriting files in `data/processed/`. Condition A is independent of
the masking seed and can be cached across runs via `a_cache`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from blindspot.conditions import ConditionConfig, build_conditions
from blindspot.explain import explain_set, grouped, importance_share, local_stability, shap_values
from blindspot.faithfulness import PROMO_FEATURE, train_injected
from blindspot.features import FEATURES, build_features
from blindspot.inventory import CO, CU, critical_ratio, order_quantity, outcomes, safety_quantiles, summarize
from blindspot.metrics import evaluate
from blindspot.model import predict, split_rows, train_final
from blindspot.whatif import SCENARIOS, whatif

PROMO_SCENARIO = "promo: activity + 20% discount"

# Maps metric key to (display_label, format_spec, worse_direction).
# Direction is one of: "high", "low", "abs" (distance from 0), or "abs1" (distance from 1).
METRICS: dict[str, tuple[str, str, str]] = {
    "wape": ("Forecast WAPE", "{:.3f}", "high"),
    "bias": ("Forecast bias (WPE)", "{:+.3f}", "abs"),
    "row_spearman_vs_A": ("Per-row SHAP agreement with A (Spearman)", "{:.3f}", "low"),
    "discount_share_change": ("Discount SHAP share, change vs A", "{:+.1%}", "abs"),
    "activity_share_change": ("Activity-flag SHAP share, change vs A", "{:+.1%}", "abs"),
    "recent_sales_share_change": ("Recent-sales SHAP share, change vs A", "{:+.1%}", "abs"),
    "promo_uplift_vs_A": ("Real-lever promo what-if, share of A's uplift", "{:.0%}", "abs1"),
    "injected_recovery": ("Known promo effect recovered (what-if / true)", "{:.0%}", "abs1"),
    "cost_per_demand": ("Everyday ordering cost / demand", "{:.3f}", "high"),
    "fill_rate": ("Everyday fill rate", "{:.1%}", "low"),
    "stockout_rate": ("Everyday stockout days", "{:.1%}", "high"),
    "promo_extra_cost": ("Extra promo-day cost from what-if error", "{:+.3f}", "high"),
}

# Metrics anchored to condition A by definition (so only B-vs-C comparisons are meaningful).
RELATIVE_TO_A = {"row_spearman_vs_A", "discount_share_change", "activity_share_change", "recent_sales_share_change",
                 "promo_uplift_vs_A"}


def worse(metric: str, x: float, y: float) -> bool:
    """Return True if value `x` is worse than `y` under `metric`'s direction."""
    d = METRICS[metric][2]
    if d == "high":
        return x > y
    if d == "low":
        return x < y
    if d == "abs":
        return abs(x) > abs(y)
    # "abs1": larger distance from 1.0 (perfect agreement or 100% recovery).
    return abs(x - 1) > abs(y - 1)


def fit_bundle(f: pd.DataFrame, cfg: dict, trainer=train_final, features: list[str] | None = None,
               resid_rows=None) -> dict:
    """Train the final model on all train rows plus a fit-only model for out-of-sample validation residuals.

    `trainer(params, n_rounds, df, features)` returns an object with `.predict`; `resid_rows(valid)` optionally
    selects which validation rows feed the residuals (default: all).
    """
    fit, valid, train, test = split_rows(f)
    final = trainer(cfg["params"], cfg["n_rounds"], train, features)
    fit_only = trainer(cfg["params"], cfg["n_rounds"], fit, features)
    v = valid if resid_rows is None else valid[resid_rows(valid)]
    return {"booster": final, "test": test, "resid": v.target.to_numpy() - predict(fit_only, v, features),
            "resid_group": v.sales_group.astype(str).to_numpy()}


def _models(cond: pd.DataFrame, panel: pd.DataFrame, c: str, cfg: dict) -> dict:
    """Model bundle for condition `c` (see `fit_bundle`)."""
    return fit_bundle(build_features(cond, panel, c), cfg)


def run_seed(panel: pd.DataFrame, mask_seed: int, cfg: dict, a_cache: dict | None = None, explain_rows: int = 2000,
             effect: float = 0.3, rate: float = 0.2, cu: float = CU, co: float = CO,
             mask_rate: float = ConditionConfig.mask_rate,
             demand_power: float = ConditionConfig.demand_power) -> pd.DataFrame:
    """Run all headline metrics for conditions A, B, and C at one masking seed.

    `mask_rate` and `demand_power` change how B and C are built (demand_power 0 = masking unrelated to demand);
    condition A does not depend on them, so its cached models stay valid.

    Returns a long-format DataFrame with columns `seed`, `condition`, `metric`, `value`.
    """
    a_cache = {} if a_cache is None else a_cache
    cc = ConditionConfig(mask_rate=mask_rate, demand_power=demand_power, seed=mask_seed)
    cond, _ = build_conditions(panel, cc)
    m = {}
    for c in "ABC":
        if c == "A" and "A" in a_cache:
            m[c] = a_cache["A"]
        else:
            m[c] = _models(cond, panel, c, cfg)
            if c == "A":
                a_cache["A"] = m[c]
    need = "BC" if "A_inj" in a_cache else "ABC"
    inj, t_inj, _, _ = train_injected(panel, cfg, rate=rate, effect=effect, cond_cfg=cc, conditions=need)
    if "A_inj" in a_cache:
        inj["A"] = a_cache["A_inj"]
    else:
        a_cache["A_inj"] = inj["A"]
    return score_models(m, inj, t_inj, mask_seed, explain_rows, effect, cu, co)


def score_models(m: dict, inj: dict, t_inj: pd.DataFrame, mask_seed: int, explain_rows: int = 2000,
                 effect: float = 0.3, cu: float = CU, co: float = CO) -> pd.DataFrame:
    """Compute every headline metric for the model bundles `m` (must include the reference "A").

    `inj` holds the matching models trained with the injected `test_promo` feature and `t_inj` their test rows.
    Returns a long-format DataFrame with columns `seed`, `condition`, `metric`, `value`.
    """
    names = list(m)
    test = m["A"]["test"]
    clean = test[test.is_clean_day == 1].copy()
    rows: list[tuple] = []

    def add(c, k, v):
        rows.append((mask_seed, c, k, float(v)))

    # Forecast accuracy on clean test days.
    for c in names:
        clean[f"pred_{c}"] = predict(m[c]["booster"], clean)
        e = evaluate(clean, f"pred_{c}").set_index("group").loc["all"]
        add(c, "wape", e.wape)
        add(c, "bias", e.wpe)

    # SHAP explanation stability and importance-share shifts.
    X = explain_set(test, n=explain_rows)
    sv = {c: shap_values(m[c]["booster"], X)[0] for c in names}
    share = {c: importance_share(sv[c]) for c in names}
    gshare = {c: grouped(sv[c]).abs().mean() for c in names}
    gshare = {c: g / g.sum() for c, g in gshare.items()}
    for c in names:
        add(c, "row_spearman_vs_A", local_stability(sv["A"], sv[c]).spearman.mean())
        add(c, "discount_share_change", share[c]["discount"] / share["A"]["discount"] - 1)
        add(c, "activity_share_change", share[c]["activity_flag"] / share["A"]["activity_flag"] - 1)
        add(c, "recent_sales_share_change", gshare[c]["recent_sales"] / gshare["A"]["recent_sales"] - 1)

    # Real-lever promotion what-if uplift relative to condition A.
    base, scen = SCENARIOS[PROMO_SCENARIO]
    up = {c: whatif(m[c]["booster"], clean, base, scen).uplift.sum() for c in names}
    for c in names:
        add(c, "promo_uplift_vs_A", up[c] / up["A"])

    # Everyday newsvendor ordering calibrated on each condition's own validation residuals.
    cr = critical_ratio(cu, co)
    g = clean.sales_group.astype(str).to_numpy()
    D = clean.sales_true.to_numpy()
    safety = {}
    for c in names:
        safety[c] = safety_quantiles(m[c]["resid"], m[c]["resid_group"], cr)
        s = summarize(outcomes(order_quantity(clean[f"pred_{c}"].to_numpy(), g, safety[c]), D, cu, co))
        add(c, "cost_per_demand", s["cost_per_demand"])
        add(c, "fill_rate", s["fill_rate"])
        add(c, "stockout_rate", s["stockout_rate"])

    # Injected promotion recovery and incremental promo-day ordering cost.
    w = t_inj[t_inj.is_clean_day == 1]
    feats = FEATURES + [PROMO_FEATURE]
    true_up = (w.demand_no_promo * effect).to_numpy()
    Dp = w.demand_no_promo.to_numpy() * (1 + effect)
    gp = w.sales_group.astype(str).to_numpy()
    for c in names:
        r = whatif(inj[c], w, {PROMO_FEATURE: 0}, {PROMO_FEATURE: 1}, feats)
        add(c, "injected_recovery", r.uplift.sum() / true_up.sum())
        cw = summarize(outcomes(order_quantity(r.scenario.to_numpy(), gp, safety[c]), Dp, cu, co))["cost_per_demand"]
        ct = summarize(outcomes(order_quantity(r.base.to_numpy() * (1 + effect), gp, safety[c]), Dp, cu, co))["cost_per_demand"]
        add(c, "promo_extra_cost", cw - ct)
    return pd.DataFrame(rows, columns=["seed", "condition", "metric", "value"])


def consistency(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize cross-seed mean/min/max per metric and count seeds where B is worse than A or C."""
    wide = results.pivot_table(index=["metric", "seed"], columns="condition", values="value")
    out = []
    for k in METRICS:
        if k not in wide.index.get_level_values(0):
            continue
        t = wide.loc[k]
        n = len(t)
        b_vs_a = np.nan if k in RELATIVE_TO_A else sum(worse(k, b, a) for a, b in zip(t.A, t.B))
        b_vs_c = sum(worse(k, b, c) for b, c in zip(t.B, t.C))
        out.append({"metric": k, "n_seeds": n, "A_mean": t.A.mean(), "B_mean": t.B.mean(), "B_min": t.B.min(),
                    "B_max": t.B.max(), "C_mean": t.C.mean(), "C_min": t.C.min(), "C_max": t.C.max(),
                    "B_worse_than_A": b_vs_a, "B_worse_than_C": b_vs_c})
    return pd.DataFrame(out)
