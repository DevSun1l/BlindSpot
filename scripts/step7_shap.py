"""Compute SHAP attributions and stability metrics for models A, B, and C on a shared test sample.

Trains three seed-varied copies of condition A (A'1..A'3) to establish a
retraining noise floor against which A-vs-B and A-vs-C explanation shifts are
evaluated. Uses exact `TreeExplainer(..., feature_perturbation="tree_path_dependent")`
to support LightGBM categorical splits without an external background dataset.

Outputs (under outputs/ unless noted):
  data/processed/shap_explain_rows.parquet, data/processed/shap_A|B|C.parquet
  step7_shap_report.md, step7_global_importance.png, step7_groups_and_stability.png
Usage:
    .\\.venv\\Scripts\\python.exe scripts\\step7_shap.py [--rows 2000]
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blindspot.data import PROJECT_ROOT  # noqa: E402
from blindspot.explain import (bootstrap_mean_ci, explain_set, global_importance, global_stability, grouped,  # noqa: E402
                               importance_share, local_stability, shap_values)
from blindspot.features import FEATURE_GROUPS, FEATURES  # noqa: E402
from blindspot.model import load_config, load_features, split_rows, train_final  # noqa: E402
from blindspot.prepare import PROCESSED_DIR  # noqa: E402

OUT = PROJECT_ROOT / "outputs"
MODELS = OUT / "models"
REF_SEEDS = [43, 44, 45]
REFS = [f"A'{i + 1}" for i in range(len(REF_SEEDS))]
COLORS = {"A": "#4C78A8", "B": "#E45756", "C": "#F58518", "A'1": "#9D9D9D"}
FOCUS = ["lag_1", "lag_7", "roll_mean_7", "roll_mean_14", "discount", "activity_flag", "weekday", "holiday_flag"]
lines: list[str] = []
problems: list[str] = []


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def check(ok: bool, msg: str) -> None:
    out(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        problems.append(msg)


def group_of(f: str) -> str:
    return next(g for g, fs in FEATURE_GROUPS.items() if f in fs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()

    cfg = load_config(MODELS / "lgb_config.json")
    boosters = {c: lgb.Booster(model_file=str(MODELS / f"lgb_{c}.txt")) for c in "ABC"}
    _, _, train_a, test = split_rows(load_features("A"))
    for name, seed in zip(REFS, REF_SEEDS):
        boosters[name] = train_final({**cfg["params"], "seed": seed}, cfg["n_rounds"], train_a)

    X = explain_set(test, n=a.rows)
    stores = X["store_id"].astype(int).to_numpy()
    sv, base, gsv, imp, share, gimp = {}, {}, {}, {}, {}, {}
    for c, b in boosters.items():
        sv[c], base[c] = shap_values(b, X)
        gsv[c] = grouped(sv[c])
        imp[c], share[c] = global_importance(sv[c]), importance_share(sv[c])
        gimp[c] = global_importance(gsv[c])

    out("# Step 7 report: SHAP explanations\n")
    out(f"- Explained rows: the same {len(X):,} clean test rows for every model "
        f"({(X.sales_group == 'low').mean():.0%} low-sales, {X.series_id.nunique():,} series, {len(set(stores)):,} stores).")
    out("- Explainer: TreeExplainer, exact, `tree_path_dependent`. It uses each model's own trees, so no background "
        "sample is needed, and it supports LightGBM's categorical splits (the interventional mode with a background "
        "sample does not). SHAP values are relative to each model's own base value (its average training prediction).")
    out(f"- Noise floor: {len(REFS)} reference models ({', '.join(REFS)}) = condition A retrained with seeds "
        f"{REF_SEEDS}. Any change smaller than A vs A' is indistinguishable from retraining noise.")
    out("- SHAP shows what each model relies on, not what causes sales.\n")

    out("## Checks\n")
    for c, b in boosters.items():
        check(np.allclose(base[c] + sv[c].sum(axis=1).to_numpy(), b.predict(X[FEATURES]), atol=1e-4),
              f"{c}: base value + sum of SHAP = model prediction (additivity)")
    p6 = X[["series_id", "dt"]].merge(pd.read_parquet(PROCESSED_DIR / "lgb_preds.parquet"), on=["series_id", "dt"], how="left")
    for c in "ABC":
        check(np.allclose(np.clip(boosters[c].predict(X[FEATURES]), 0, None), p6[f"lgb_{c}"]),
              f"{c}: explained model reproduces the Step 6 test forecasts")
    ties = np.mean([[len(np.unique(r)) < len(r) for r in sv[c].abs().to_numpy()] for c in "ABC"])
    check(ties == 0, "no tied |SHAP| values within a row (rank-based stability measures are well defined)")

    # 1. Global feature importance
    out("\n## 1. Global importance\n")
    out("Raw = mean |SHAP| in sales units. Share = raw / sum over all features (scale-free). Use **share** to compare "
        f"models: B forecasts lower overall, so its SHAP values are smaller everywhere (mean total |SHAP| per row: "
        + ", ".join(f"{c} {sv[c].abs().sum(axis=1).mean():.3f}" for c in ["A", "B", "C", "A'1"]) + ").\n")
    out("| rank (A) | feature | group | raw A | raw B | raw C | share A | share B | share C | share A' (mean) |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    order = imp["A"].index
    for i, f in enumerate(order, 1):
        ref = np.mean([share[r][f] for r in REFS])
        out(f"| {i} | {f} | {group_of(f)} | {imp['A'][f]:.4f} | {imp['B'][f]:.4f} | {imp['C'][f]:.4f} | "
            f"{share['A'][f]:.1%} | {share['B'][f]:.1%} | {share['C'][f]:.1%} | {ref:.1%} |")
    out("")
    for c in ["A", "B", "C", "A'1"]:
        out(f"- Top 5 in {c}: {', '.join(imp[c].index[:5])}")

    # 2. Feature group importance shares
    out("\n## 2. Feature groups (share of mean |group SHAP|)\n")
    gshare = {c: gimp[c] / gimp[c].sum() for c in gimp}
    out("| group | A | B | C | A' (mean) |")
    out("|---|---|---|---|---|")
    for g in FEATURE_GROUPS:
        out(f"| {g} | {gshare['A'][g]:.1%} | {gshare['B'][g]:.1%} | {gshare['C'][g]:.1%} | "
            f"{np.mean([gshare[r][g] for r in REFS]):.1%} |")
    same_order = all(list(gimp[c].index) == list(gimp["A"].index) for c in gimp)
    out(f"\nGroup ranking identical in every model: {same_order}. (With only 5 groups a rank correlation is too coarse "
        "to be informative, so groups are compared by share only.)")

    # 3. Global ranking stability
    out("\n## 3. Stability of the global feature ranking (22 features)\n")
    out("| comparison | Spearman | Kendall | top-5 overlap |")
    out("|---|---|---|---|")
    refstab = [global_stability(imp["A"], imp[r]) for r in REFS]
    out(f"| A vs A' (mean of {len(REFS)}) | {np.mean([s['spearman'] for s in refstab]):.3f} | "
        f"{np.mean([s['kendall'] for s in refstab]):.3f} | {np.mean([s['top5_overlap'] for s in refstab]):.0%} |")
    for x, y in [("A", "B"), ("A", "C"), ("B", "C")]:
        s = global_stability(imp[x], imp[y])
        out(f"| {x} vs {y} | {s['spearman']:.3f} | {s['kendall']:.3f} | {s['top5_overlap']:.0%} |")

    # 4. Per-row explanation stability
    out("\n## 4. Stability of per-row explanations (same row, two models)\n")
    out("| comparison | mean per-row Spearman (95% CI) | mean top-3 overlap | rows with identical top-3 |")
    out("|---|---|---|---|")
    loc = {("A", r): local_stability(sv["A"], sv[r]) for r in REFS}
    ref_rho = np.mean([loc[("A", r)].spearman.to_numpy() for r in REFS], axis=0)
    ref_top = np.mean([loc[("A", r)].top3_overlap.to_numpy() for r in REFS], axis=0)
    ref_same = np.mean([(loc[("A", r)].top3_overlap == 1).mean() for r in REFS])
    lo, hi = bootstrap_mean_ci(ref_rho, stores)
    out(f"| A vs A' (mean of {len(REFS)}) | {ref_rho.mean():.3f} ({lo:.3f}-{hi:.3f}) | {ref_top.mean():.2f} | {ref_same:.0%} |")
    for x, y in [("A", "B"), ("A", "C"), ("B", "C")]:
        loc[(x, y)] = local_stability(sv[x], sv[y])
        lo, hi = bootstrap_mean_ci(loc[(x, y)].spearman.to_numpy(), stores)
        out(f"| {x} vs {y} | {loc[(x, y)].spearman.mean():.3f} ({lo:.3f}-{hi:.3f}) | {loc[(x, y)].top3_overlap.mean():.2f} | "
            f"{(loc[(x, y)].top3_overlap == 1).mean():.0%} |")
    d = loc[("A", "C")].spearman.to_numpy() - loc[("A", "B")].spearman.to_numpy()
    lo, hi = bootstrap_mean_ci(d, stores)
    out(f"\nPaired test, same rows: (A vs C) minus (A vs B) agreement = {d.mean():+.3f} (95% CI {lo:+.3f} to {hi:+.3f}); "
        f"censoring disagrees with A more than noise on {(d > 0).mean():.0%} of rows. "
        + ("The CI excludes 0: censoring changes explanations significantly more than matched noise."
           if lo > 0 else "The CI includes 0: no significant difference between censoring and noise."))

    # 5. Relative importance-share drift on focus features
    out("\n## 5. Drift of focus features: change in importance share vs A\n")
    out("| feature | B vs A | C vs A | A' vs A (range over seeds) |")
    out("|---|---|---|---|")
    for f in FOCUS:
        refs = [share[r][f] / share["A"][f] - 1 for r in REFS]
        out(f"| {f} | {share['B'][f] / share['A'][f] - 1:+.0%} | {share['C'][f] / share['A'][f] - 1:+.0%} | "
            f"{min(refs):+.0%} to {max(refs):+.0%} |")

    out("\n## 6. Promotion levers: signed contribution (sales units) and relative to the model's base value\n")
    promo, act = (X.discount < 1).to_numpy(), (X.activity_flag == 1).to_numpy()
    out(f"Discounted rows: {promo.sum():,}; activity rows: {act.sum():,}.\n")
    out("| model | base value | discount SHAP on discounted rows | as % of base | activity_flag SHAP on activity rows | as % of base |")
    out("|---|---|---|---|---|---|")
    for c in ["A", "B", "C"] + REFS:
        dsc, acf = sv[c].discount[promo].mean(), sv[c].activity_flag[act].mean()
        out(f"| {c} | {base[c]:.3f} | {dsc:+.4f} | {dsc / base[c]:+.1%} | {acf:+.4f} | {acf / base[c]:+.1%} |")

    # 7. Forecast accuracy change vs explanation shift
    res = pd.read_csv(OUT / "step6_results.csv").query("group=='all'").set_index("model")
    wa, wb, wc = (res.loc[m, "wape"] for m in ["LightGBM A (clean)", "LightGBM B (censored)", "LightGBM C (noise)"])
    out("\n## 7. Accuracy change vs explanation change\n")
    out("| comparison | WAPE change | explanation change (1 - per-row Spearman) | x noise floor |")
    out("|---|---|---|---|")
    floor = 1 - ref_rho.mean()
    for lbl, wchg, rho in [("A vs B", f"{wb / wa - 1:+.1%}", loc[("A", "B")].spearman.mean()),
                           ("A vs C", f"{wc / wa - 1:+.1%}", loc[("A", "C")].spearman.mean())]:
        out(f"| {lbl} | {wchg} | {1 - rho:.3f} | {(1 - rho) / floor:.1f}x |")
    out(f"| A vs A' | (retrain only) | {floor:.3f} | 1.0x |")

    # Save global importance and group/stability figures.
    top = list(order[:12])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    w = 0.2
    for i, c in enumerate(["A", "B", "C", "A'1"]):
        ax.barh(np.arange(len(top)) + (i - 1.5) * w, share[c][top].values, height=w, label=c, color=COLORS[c])
    ax.set_yticks(range(len(top)), top)
    ax.invert_yaxis()
    ax.set(title="Importance share (mean |SHAP| / total), top 12 features by model A", xlabel="share of total |SHAP|")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "step7_global_importance.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    gs_ = list(FEATURE_GROUPS)
    for i, c in enumerate(["A", "B", "C", "A'1"]):
        ax[0].bar(np.arange(len(gs_)) + (i - 1.5) * w, [gshare[c][g] for g in gs_], width=w, label=c, color=COLORS[c])
    ax[0].set_xticks(range(len(gs_)), gs_)
    ax[0].set(title="Share of explanation by feature group", ylabel="share of mean |group SHAP|")
    ax[0].legend()
    data = [ref_rho] + [loc[p].spearman.to_numpy() for p in [("A", "B"), ("A", "C"), ("B", "C")]]
    ax[1].boxplot(data, tick_labels=["A vs A'", "A vs B", "A vs C", "B vs C"], showfliers=False)
    ax[1].set(title="Per-row agreement of explanations", ylabel="Spearman rank correlation of |SHAP|")
    fig.tight_layout()
    fig.savefig(OUT / "step7_groups_and_stability.png", dpi=110)
    plt.close(fig)
    out("\n![global](step7_global_importance.png)\n\n![groups](step7_groups_and_stability.png)")

    X.to_parquet(PROCESSED_DIR / "shap_explain_rows.parquet", index=False)
    for c in "ABC":
        keys = X[["series_id", "dt"]].reset_index(drop=True)
        pd.concat([keys, sv[c].reset_index(drop=True)], axis=1).assign(base_value=base[c]).to_parquet(
            PROCESSED_DIR / f"shap_{c}.parquet", index=False)
    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    report = OUT / "step7_shap_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
