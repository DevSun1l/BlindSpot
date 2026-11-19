"""Interactive Streamlit app for comparing models A (clean), B (censored), and C (matched noise).

Run:
    .\\.venv\\Scripts\\streamlit.exe run app\\streamlit_app.py

Reads precomputed pipeline artifacts and trained boosters from disk without retraining.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from blindspot.app_data import (COLORS, CONDITIONS, cost_curve, explain_row, global_shares, global_stability,
                                history, load_assets, order_table, row_stability, safety_for, series_options,
                                showcase_series, top_drivers)
from blindspot.experiment import METRICS, RELATIVE_TO_A
from blindspot.features import FEATURES
from blindspot.whatif import LEVER_LIMITS, whatif

st.set_page_config(page_title="BlindSpot", layout="wide")
HALF_DAY = timedelta(hours=12)


@st.cache_resource(show_spinner="Loading data and models ...")
def assets() -> dict:
    a = load_assets()
    a["explainers"] = {c: shap.TreeExplainer(b, feature_perturbation="tree_path_dependent")
                       for c, b in a["boosters"].items()}
    a["options"] = series_options(a["test"])
    return a


A = assets()
opts = A["options"]

# Sidebar: series, forecast-day, and relative newsvendor cost controls.
st.sidebar.title("BlindSpot")
st.sidebar.caption("How stockout censoring changes forecasts, explanations and orders.")
only_promo = st.sidebar.checkbox("Only series with a promotion in the test week", value=False)
pool = opts[opts.promo_days > 0] if only_promo else opts
default_sid = showcase_series(opts)
stores = sorted(pool.store_id.unique())
default_store = int(opts.loc[opts.series_id == default_sid, "store_id"].iloc[0])
store = st.sidebar.selectbox("Store", stores, index=stores.index(default_store) if default_store in stores else 0)
prods = pool[pool.store_id == store].sort_values("product_id")
labels = {int(r.series_id): f"product {r.product_id} ({r.sales_group} demand, {r.promo_days} promo days)"
          for r in prods.itertuples()}
sids = list(labels)
sid = st.sidebar.selectbox("Product", sids, format_func=lambda s: labels.get(int(s), str(s)),
                           index=sids.index(default_sid) if default_sid in sids else 0)

t = A["test"][A["test"].series_id == sid].sort_values("dt").reset_index(drop=True)
day_labels = {i: f"{r['dt']:%a %d %b}" + ("" if r.is_clean_day else " (stockout day)") + (" - promo" if r.activity_flag else "")
              for i, r in t.iterrows()}
clean_idx = [i for i in day_labels if t.loc[i, "is_clean_day"] == 1]
day = st.sidebar.selectbox("Forecast day (test week)", list(day_labels), format_func=lambda i: day_labels.get(int(i), str(i)),
                           index=list(day_labels).index(clean_idx[0]) if clean_idx else 0)
row = t.iloc[[day]]

st.sidebar.markdown("**Costs (relative)**")
cu = st.sidebar.number_input("Cost of one lost sale", 0.1, 10.0, 1.0, 0.1)
co = st.sidebar.number_input("Cost of one wasted unit", 0.05, 10.0, 0.5, 0.05)
st.sidebar.caption(f"Critical ratio = {cu / (cu + co):.2f}: the order aims to cover demand on about "
                   f"{cu / (cu + co):.0%} of days.")

r0 = row.iloc[0]
clean_day = bool(r0.is_clean_day)
true_d = float(r0.sales_true) if clean_day else None
group = str(r0.sales_group)

st.title(f"Store {store}, product {int(r0.product_id)} - {r0['dt']:%A %d %B %Y}")
st.caption("Three copies of the same LightGBM model, trained on: **A** recorded sales on stockout-free days, "
           "**B** the same days with 30% of them censored by a simulated stockout, **C** the same amount of "
           "distortion with a random sign. All three see identical inputs for the test week.")

tab1, tab2, tab3 = st.tabs(["Forecast, explanation and order", "Comparison across models", "Summary of results"])


def fig_layout(fig: go.Figure, h: int = 320, **kw) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.3), **kw)
    return fig


# Tab 1: single-series forecast, SHAP attribution, promotion what-if, and order quantity.
with tab1:
    # Headline demand and per-model forecast metrics.
    cols = st.columns(4)
    cols[0].metric("True demand", f"{true_d:.2f}" if clean_day else "unknown",
                   help=None if clean_day else "A real stockout happened that day, so sales understate demand.")
    for i, c in enumerate("ABC", 1):
        f = float(r0[f"pred_{c}"])
        cols[i].metric(f"Forecast {CONDITIONS[c]}", f"{f:.2f}",
                       delta=f"{f - true_d:+.2f} vs true" if clean_day else None, delta_color="off")
    if not clean_day:
        st.info("This was a real stockout day: recorded sales are a lower bound, so forecasts are not scored here.")

    # Historical sales, simulated/real stockout markers, and test-week forecasts.
    h = history(A["hist"], sid)
    fig = go.Figure()
    fig.add_scatter(x=h.dt, y=h.sales_true, mode="lines+markers", name="recorded sales (A's training data)",
                    line=dict(color=COLORS["A"]), marker=dict(size=5))
    mb = h[(h.is_masked == 1)]
    fig.add_scatter(x=mb.dt, y=mb.sales_B, mode="markers", name="what B saw (censored day)",
                    marker=dict(color=COLORS["B"], size=9, symbol="triangle-down"))
    so = h[h.is_stockout_day == 1]
    fig.add_scatter(x=so.dt, y=so.sales_true, mode="markers", name="real stockout day",
                    marker=dict(color="#7f7f7f", size=9, symbol="x"))
    for c in "ABC":
        fig.add_scatter(x=t.dt, y=t[f"pred_{c}"], mode="lines+markers", name=f"forecast {c}",
                        line=dict(color=COLORS[c], dash="dot"), marker=dict(size=6))
    first_test = t.dt.min()
    fig.add_vline(x=pd.Timestamp(first_test).to_pydatetime() - HALF_DAY, line=dict(color="#999", dash="dash"))
    fig.add_vrect(x0=pd.Timestamp(r0["dt"]).to_pydatetime() - HALF_DAY,
                  x1=pd.Timestamp(r0["dt"]).to_pydatetime() + HALF_DAY, fillcolor="#999",
                  opacity=0.15, line_width=0)
    st.subheader("Sales history and forecasts")
    st.plotly_chart(fig_layout(fig, 340, yaxis_title="units (normalised)"), width="stretch")
    st.caption("Left of the dashed line: last 6 training weeks. Right: test week with each model's next-day forecast. "
               "Shaded: the selected day. Triangles show how much B was told was sold on its censored days.")

    # Local SHAP attribution comparison for the selected forecast day.
    st.subheader("Why each model forecasts what it does (SHAP)")
    sv, base = explain_row(A["explainers"], row)
    grouped = st.toggle("Group related features (recent sales, calendar, promotion, weather, identity)", value=False)
    top = top_drivers(sv, k=8, grouped=grouped)
    fig = go.Figure()
    for c in "ABC":
        fig.add_bar(y=top.index, x=top[c], orientation="h", name=CONDITIONS[c], marker_color=COLORS[c])
    fig.add_vline(x=0, line_color="#999")
    st.plotly_chart(fig_layout(fig, 60 + 34 * len(top), barmode="group", xaxis_title="push on the forecast (units)"),
                    width="stretch")
    stab = row_stability(sv)
    st.caption(f"Base values (average forecast): A {base['A']:.2f}, B {base['B']:.2f}, C {base['C']:.2f}; "
               "forecast = base + sum of all bars. For this forecast, agreement with A's ranking of all 22 features: "
               + ", ".join(f"{r.comparison} Spearman {r.spearman:.2f}, top-3 overlap {r.top3_overlap:.0%}"
                           for r in stab.itertuples())
               + ". SHAP shows what each model relies on, not what causes sales.")

    # Promotion what-if controls restricted to planned levers within training support.
    st.subheader("What-if: change tomorrow's promotion plan")
    st.caption("Only the planned levers can change. Sales history, calendar, weather and product are fixed, so the "
               "scenario cannot rewrite the past. Discount is limited to the range seen in training.")
    lo, hi = LEVER_LIMITS["discount"]
    st.caption(f"Actual plan for this day: price multiplier {r0.discount:.2f}, promotion event "
               f"{'on' if r0.activity_flag else 'off'}.")
    # An explicit switch decides whether the user's plan is used; slider positions alone never do, so a
    # deliberately chosen value is always used exactly and an untouched page always shows the actual plan.
    changed = st.toggle("Try a different plan", value=False, key=f"try_{sid}_{day}")
    w1, w2 = st.columns([3, 1])
    start = float(np.clip(round(r0.discount * 20) / 20, lo, hi))        # nearest 0.05 step inside the allowed range
    disc = w1.slider("Price multiplier (1.00 = no discount, 0.80 = 20% off)", float(lo), float(hi), start, 0.05,
                     key=f"disc_{sid}_{day}", disabled=not changed)
    act = w2.toggle("Promotion event (activity_flag)", value=bool(r0.activity_flag), key=f"act_{sid}_{day}",
                    disabled=not changed)
    # Empty overrides keep the row's observed promotion values as the baseline.
    plan = {"discount": disc, "activity_flag": int(act)} if changed else {}
    wi = {c: whatif(A["boosters"][c], row, {}, plan).iloc[0] for c in "ABC"}
    wtab = pd.DataFrame([{"model": CONDITIONS[c], "forecast with actual plan": wi[c].base,
                          "forecast with your plan": wi[c].scenario, "uplift (units)": wi[c].uplift,
                          "uplift (%)": wi[c].uplift / wi[c].base if wi[c].base > 0 else np.nan} for c in "ABC"])
    st.dataframe(wtab.style.format({"forecast with actual plan": "{:.2f}", "forecast with your plan": "{:.2f}",
                                    "uplift (units)": "{:+.2f}", "uplift (%)": "{:+.0%}"}),
                 hide_index=True, width="stretch")
    ua, ub = wi["A"].uplift, wi["B"].uplift
    meaningful = abs(ua) >= max(0.1, 0.05 * wi["A"].base) and np.sign(ua) == np.sign(ub)
    if changed and meaningful:
        st.caption(f"For this plan B's change is {ub / ua:.0%} of A's. ")
    elif changed:
        st.caption("For this single day the change is small or the models disagree in direction, so a ratio "
                   "would not mean much.")
    st.caption("Across all test days, B's promotion uplift was about 61% of A's, and on a known injected effect "
               "the clean model recovered about 90% against about 59% for the censored model (Step 9).")

    # Newsvendor order recommendation and demand-cost curve under the selected plan.
    st.subheader("Recommended order")
    safety = safety_for(A["val_errors"], cu, co)
    fc = {c: float(wi[c].scenario) for c in "ABC"}
    known = true_d if (clean_day and not changed) else None
    ot = order_table(fc, group, safety, cu, co, known)
    fmt = {k: "{:.2f}" for k in ot.columns if k != "model"}
    st.dataframe(ot.style.format(fmt), hide_index=True, width="stretch")
    if changed:
        st.caption("Your plan differs from what actually happened, so the true demand under it is unknown; "
                   "the chart shows the cost for any demand.")
    st.caption(f"Order = forecast with your plan + safety allowance. The allowance is the {cu / (cu + co):.0%} "
               f"quantile of each model's own past forecast errors for {group}-demand products. B measures its "
               "errors on censored sales, so its allowance is also too small.")
    orders = dict(zip("ABC", ot["recommended order"]))
    dmax = max(max(orders.values()), true_d or 0, 0.5) * 1.8
    cc = cost_curve(orders, dmax, cu, co)
    fig = go.Figure()
    for c in "ABC":
        fig.add_scatter(x=cc.demand, y=cc[c], mode="lines", name=f"order of {c} ({orders[c]:.2f})",
                        line=dict(color=COLORS[c]))
    if known is not None:
        fig.add_vline(x=known, line=dict(color="#333", dash="dash"), annotation_text="true demand")
    st.plotly_chart(fig_layout(fig, 300, xaxis_title="demand that actually turns up", yaxis_title="cost"),
                    width="stretch")
    st.caption("Each line: the cost of that model's order if demand turned out to be x. Left of an order's minimum "
               "you waste stock; right of it you lose sales (which costs more per unit).")

# Tab 2: global SHAP importance shares, stability, and faithfulness summary.
with tab2:
    st.subheader("Which features each model relies on (all 2,000 explained test rows)")
    g2 = st.toggle("Group related features", value=True, key="g2")
    sh = global_shares(A["shap_global"], grouped=g2)
    fig = go.Figure()
    for c in "ABC":
        fig.add_bar(y=sh.index[::-1], x=sh[c][::-1], orientation="h", name=CONDITIONS[c], marker_color=COLORS[c])
    st.plotly_chart(fig_layout(fig, int(80 + 34 * len(sh) * (1 if g2 else 0.8)), barmode="group",
                               xaxis_title="share of total importance (mean |SHAP|)", xaxis_tickformat=".0%"),
                    width="stretch")
    st.caption("Shares are scale-free: B forecasts lower overall, so its raw SHAP values are smaller everywhere.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stability of explanations** (same rows, two models)")
        gs = global_stability(A["shap_global"])
        st.dataframe(gs.style.format({"mean per-row Spearman": "{:.3f}", "mean top-3 overlap": "{:.2f}"}),
                     hide_index=True, width="stretch")
        if A["noise_floor"]:
            st.caption(f"Noise floor: retraining A with a different random seed gives {A['noise_floor']:.3f}. "
                       "Lower than that means the training data, not chance, changed the explanation.")
    with c2:
        st.markdown("**Faithfulness** (Step 8)")
        fa = A["faithfulness"]
        if fa is not None:
            fa = fa.rename(columns={"model": "model", "deletion_gain": "deletion gain vs random",
                                    "insertion_gain": "insertion gain vs random",
                                    "whatif_recovery": "known promo effect recovered (what-if)",
                                    "shap_recovery": "known promo effect recovered (SHAP)",
                                    "promo_rank": "promo feature rank"})
            st.dataframe(fa.style.format({"deletion gain vs random": "{:+.3f}", "insertion gain vs random": "{:+.3f}",
                                          "known promo effect recovered (what-if)": "{:.0%}",
                                          "known promo effect recovered (SHAP)": "{:.0%}"}),
                         hide_index=True, width="stretch")
            st.caption("SHAP describes each model faithfully (similar gains), but B has learned a promotion effect "
                       "that is far too small, while still ranking it 2nd: the chart looks the same, the size is wrong.")

# Tab 3: cross-seed headline metrics table and takeaways.
with tab3:
    st.subheader("Results across the three conditions")
    sr, rb = A["seed_results"], A["robustness"]
    if sr is None or rb is None:
        st.warning("Run Step 11 (scripts/step11_experiment.py) to produce the results table.")
    else:
        main_seed = 42 if 42 in sr.seed.values else int(sr.seed.min())
        main = sr[sr.seed == main_seed].pivot_table(index="metric", columns="condition", values="value")
        n = int(rb.n_seeds.iloc[0])
        rows = []
        for r in rb.itertuples():
            f = METRICS[r.metric][1]
            rows.append({"metric": METRICS[r.metric][0],
                         "A": "reference" if r.metric in RELATIVE_TO_A else f.format(main.loc[r.metric, "A"]),
                         "B": f.format(main.loc[r.metric, "B"]), "C": f.format(main.loc[r.metric, "C"]),
                         f"B range over {n} seeds": f"{f.format(r.B_min)} to {f.format(r.B_max)}",
                         "B worse than A": "n/a" if pd.isna(r.B_worse_than_A) else f"{int(r.B_worse_than_A)}/{n}",
                         "B worse than C": f"{int(r.B_worse_than_C)}/{n}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=36 * (len(rows) + 1) + 4)
        st.caption(f"Values for masking seed {main_seed}; ranges and counts over {n} masking seeds (Step 11). Ordering costs are "
                   "per unit of true demand with lost sale = 1 and wasted unit = 0.5.")
    st.markdown(
        "**What this shows**\n"
        "- Censoring (B) makes forecasts lower and less accurate, but WAPE rises only about 10%.\n"
        "- Its explanations change more than with matched noise (C), mainly for the promotion features.\n"
        "- SHAP still describes B faithfully; it is B's learned promotion effect that is too small.\n"
        "- Ordering on B's forecasts and what-ifs causes more stockouts and higher cost.\n\n"
        "**Limits**: one dataset (5,000 sampled series), one model type, simulated censoring, assumed costs, "
        "and a 7-day test week. SHAP and what-if describe the models, not customer behaviour.")
