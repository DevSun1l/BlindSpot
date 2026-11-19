"""Step 11: the whole A/B/C experiment for one masking seed, computed in memory.

Nothing is written to data/processed, so the main (seed 42) files from Steps 3-10 are never overwritten.
Settings are the frozen ones from the earlier steps: LightGBM config from Step 6a, the same explained rows as
Step 7, the same promo scenario as Step 9, the same newsvendor rule and costs as Step 10.

Condition A does not depend on the masking seed (it is the recorded data), so its models are computed once
and reused through `a_cache`.
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

# metric key -> (label, unit/format, which direction is worse: "high", "low" or "abs")
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


# Metrics defined relative to A: A is the reference by construction, so "B worse than A" is trivially true there;
# only the B-vs-C comparison is informative.
RELATIVE_TO_A = {"row_spearman_vs_A", "discount_share_change", "activity_share_change", "recent_sales_share_change",
                 "promo_uplift_vs_A"}


def worse(metric: str, x: float, y: float) -> bool:
    """True if value x is worse than y for this metric."""
    d = METRICS[metric][2]
    if d == "high":
        return x > y
    if d == "low":
        return x < y
    if d == "abs":
        return abs(x) > abs(y)
    return abs(x - 1) > abs(y - 1)          # "abs1": distance from 1 (perfect agreement / recovery)


def _models(cond: pd.DataFrame, panel: pd.DataFrame, c: str, cfg: dict) -> dict:
    f = build_features(cond, panel, c)
    fit, valid, train, test = split_rows(f)
    final = train_final(cfg["params"], cfg["n_rounds"], train)
    fit_only = train_final(cfg["params"], cfg["n_rounds"], fit)          # out-of-sample validation errors
    return {"booster": final, "test": test, "resid": valid.target.to_numpy() - predict(fit_only, valid),
            "resid_group": valid.sales_group.astype(str).to_numpy()}


def run_seed(panel: pd.DataFrame, mask_seed: int, cfg: dict, a_cache: dict | None = None, explain_rows: int = 2000,
             effect: float = 0.3, rate: float = 0.2, cu: float = CU, co: float = CO) -> pd.DataFrame:
    """All headline metrics for conditions A, B, C at one masking seed. Returns long format:
    columns seed, condition, metric, value."""
    a_cache = {} if a_cache is None else a_cache
    cc = ConditionConfig(seed=mask_seed)
    cond, _ = build_conditions(panel, cc)
    m = {}
    for c in "ABC":
        if c == "A" and "A" in a_cache:
            m[c] = a_cache["A"]
        else:
            m[c] = _models(cond, panel, c, cfg)
            if c == "A":
                a_cache["A"] = m[c]
    test = m["A"]["test"]
    clean = test[test.is_clean_day == 1].copy()
    rows: list[tuple] = []

    def add(c, k, v):
        rows.append((mask_seed, c, k, float(v)))

    # forecast accuracy
    for c in "ABC":
        clean[f"pred_{c}"] = predict(m[c]["booster"], clean)
        e = evaluate(clean, f"pred_{c}").set_index("group").loc["all"]
        add(c, "wape", e.wape)
        add(c, "bias", e.wpe)

    # explanations (same rows as Step 7)
    X = explain_set(test, n=explain_rows)
    sv = {c: shap_values(m[c]["booster"], X)[0] for c in "ABC"}
    share = {c: importance_share(sv[c]) for c in "ABC"}
    gshare = {c: grouped(sv[c]).abs().mean() for c in "ABC"}
    gshare = {c: g / g.sum() for c, g in gshare.items()}
    for c in "ABC":
        add(c, "row_spearman_vs_A", local_stability(sv["A"], sv[c]).spearman.mean())
        add(c, "discount_share_change", share[c]["discount"] / share["A"]["discount"] - 1)
        add(c, "activity_share_change", share[c]["activity_flag"] / share["A"]["activity_flag"] - 1)
        add(c, "recent_sales_share_change", gshare[c]["recent_sales"] / gshare["A"]["recent_sales"] - 1)

    # what-if on real levers (Step 9 promo scenario)
    base, scen = SCENARIOS[PROMO_SCENARIO]
    up = {c: whatif(m[c]["booster"], clean, base, scen).uplift.sum() for c in "ABC"}
    for c in "ABC":
        add(c, "promo_uplift_vs_A", up[c] / up["A"])

    # everyday ordering (Step 10 part 1)
    cr = critical_ratio(cu, co)
    g = clean.sales_group.astype(str).to_numpy()
    D = clean.sales_true.to_numpy()
    safety = {}
    for c in "ABC":
        safety[c] = safety_quantiles(m[c]["resid"], m[c]["resid_group"], cr)
        s = summarize(outcomes(order_quantity(clean[f"pred_{c}"].to_numpy(), g, safety[c]), D, cu, co))
        add(c, "cost_per_demand", s["cost_per_demand"])
        add(c, "fill_rate", s["fill_rate"])
        add(c, "stockout_rate", s["stockout_rate"])

    # injected promo: recovery (Step 9 part 2) and promo-day ordering cost (Step 10 part 2)
    need = "BC" if "A_inj" in a_cache else "ABC"
    inj, t_inj, _, _ = train_injected(panel, cfg, rate=rate, effect=effect, cond_cfg=cc, conditions=need)
    if "A_inj" in a_cache:
        inj["A"] = a_cache["A_inj"]
    else:
        a_cache["A_inj"] = inj["A"]
    w = t_inj[t_inj.is_clean_day == 1]
    feats = FEATURES + [PROMO_FEATURE]
    true_up = (w.demand_no_promo * effect).to_numpy()
    Dp = w.demand_no_promo.to_numpy() * (1 + effect)
    gp = w.sales_group.astype(str).to_numpy()
    for c in "ABC":
        r = whatif(inj[c], w, {PROMO_FEATURE: 0}, {PROMO_FEATURE: 1}, feats)
        add(c, "injected_recovery", r.uplift.sum() / true_up.sum())
        cw = summarize(outcomes(order_quantity(r.scenario.to_numpy(), gp, safety[c]), Dp, cu, co))["cost_per_demand"]
        ct = summarize(outcomes(order_quantity(r.base.to_numpy() * (1 + effect), gp, safety[c]), Dp, cu, co))["cost_per_demand"]
        add(c, "promo_extra_cost", cw - ct)
    return pd.DataFrame(rows, columns=["seed", "condition", "metric", "value"])


def consistency(results: pd.DataFrame) -> pd.DataFrame:
    """Per metric: seed-42-style first-seed values, mean/min/max over seeds for B and C, and in how many seeds
    B is worse than A and worse than C."""
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
