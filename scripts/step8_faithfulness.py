"""Evaluate SHAP faithfulness via deletion/insertion curves and synthetic promotion recovery.

Part 1: Deletion and insertion curves for models A, B, and C on the explained
        test sample, comparing |SHAP| ordering against random orderings.
Part 2: Injected synthetic promotion (`test_promo`) with a known uplift,
        measuring what-if and SHAP recovery across conditions A, B, and C.

Outputs (outputs/): step8_faithfulness_report.md, step8_results.csv,
                    step8_curves.png, step8_promo_recovery.png
Usage:
    .\\.venv\\Scripts\\python.exe scripts\\step8_faithfulness.py [--rows 500] [--effect 0.3] [--rate 0.2]
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
from blindspot.explain import bootstrap_mean_ci, shap_values  # noqa: E402
from blindspot.faithfulness import (PROMO_FEATURE, deletion_insertion, reference_values, train_injected,  # noqa: E402
                                    whatif_uplift)
from blindspot.features import FEATURES  # noqa: E402
from blindspot.metrics import evaluate  # noqa: E402
from blindspot.model import load_config, load_features, predict, split_rows  # noqa: E402
from blindspot.prepare import PROCESSED_DIR, load_panel  # noqa: E402

OUT = PROJECT_ROOT / "outputs"
MODELS = OUT / "models"
COLORS = {"A": "#4C78A8", "B": "#E45756", "C": "#F58518"}
lines: list[str] = []
problems: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def check(ok: bool, msg: str) -> None:
    out(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        problems.append(msg)


def part1(rows: int) -> dict:
    out("## Part 1: deletion and insertion curves\n")
    X_all = pd.read_parquet(PROCESSED_DIR / "shap_explain_rows.parquet")
    X = X_all.sample(n=min(rows, len(X_all)), random_state=0).sort_index()
    stores = X["store_id"].astype(int).to_numpy()
    _, _, train_a, _ = split_rows(load_features("A"))
    ref = reference_values(train_a, FEATURES)
    out(f"{len(X):,} of Step 7's explained rows. Removed features are set to condition A's training median "
        "(numeric) or most frequent value (categorical). Random order = average of 5 random orders. "
        "Score = area under the normalised curve (0-1, higher = forecast moves faster = explanation more faithful).\n")
    out("| model | rows used | deletion: SHAP order | deletion: random | gain (95% CI) | insertion: SHAP order | insertion: random | gain (95% CI) |")
    out("|---|---|---|---|---|---|---|---|")
    curves = {}
    for c in "ABC":
        b = lgb.Booster(model_file=str(MODELS / f"lgb_{c}.txt"))
        sv = pd.read_parquet(PROCESSED_DIR / f"shap_{c}.parquet").iloc[X.index]
        sv.index = X.index
        r = deletion_insertion(b, X, sv, ref, FEATURES)
        st = stores[X.index.get_indexer(r.index)]
        gd, gi = (r.del_shap - r.del_random).to_numpy(), (r.ins_shap - r.ins_random).to_numpy()
        (dlo, dhi), (ilo, ihi) = bootstrap_mean_ci(gd, st), bootstrap_mean_ci(gi, st)
        out(f"| {c} | {len(r):,} | {r.del_shap.mean():.3f} | {r.del_random.mean():.3f} | {gd.mean():+.3f} ({dlo:+.3f} to {dhi:+.3f}) | "
            f"{r.ins_shap.mean():.3f} | {r.ins_random.mean():.3f} | {gi.mean():+.3f} ({ilo:+.3f} to {ihi:+.3f}) |")
        curves[c] = (r.attrs["curves"], gd.mean(), gi.mean(), dlo, ilo)
    out("")
    for c, (_, gd, gi, dlo, ilo) in curves.items():
        check(dlo > 0 and ilo > 0, f"{c}: SHAP order beats random order in both deletion and insertion (CI above 0)")
    return curves


def part2(effect: float, rate: float, cfg: dict) -> dict:
    out(f"\n## Part 2: injected promotion with a known effect (+{effect:.0%} on {rate:.0%} of days)\n")
    boosters, test, cond, pi = train_injected(load_panel(sample_only=True), cfg, rate=rate, effect=effect)
    feats = FEATURES + [PROMO_FEATURE]
    flag = pi[["series_id", "dt", PROMO_FEATURE, "demand_no_promo"]]

    tr = (cond.split == "train") & (cond.is_clean_day == 1)
    m_on = cond.loc[tr].merge(flag, on=["series_id", "dt"])
    out(f"- Masking in condition B: {m_on.loc[m_on[PROMO_FEATURE] == 1, 'is_masked'].mean():.1%} of injected-promo clean "
        f"train days vs {m_on.loc[m_on[PROMO_FEATURE] == 0, 'is_masked'].mean():.1%} of other clean train days "
        "(demand-linked masking hits promo days harder, as real stockouts would).")

    res = {}
    clean = test[test.is_clean_day == 1].copy()
    for c in "ABC":
        clean[f"pred_{c}"] = predict(boosters[c], clean, feats)

    on = clean[clean[PROMO_FEATURE] == 1]
    true_uplift = (on.demand_no_promo * effect).to_numpy()
    out(f"- Test rows: {len(clean):,} clean test days, of which {len(on):,} have the injected promo. "
        f"True mean uplift on those rows: {true_uplift.mean():.4f} units (= {effect:.0%} of demand without promo).\n")

    X = on.sample(n=min(2000, len(on)), random_state=0)
    out("What-if = forecast with test_promo=1 minus forecast with test_promo=0, compared with the true uplift. "
        f"SHAP is relative to the model's average forecast, which already includes promo days ({rate:.0%}), so its "
        f"yardstick is (1 - {rate:.0%}) x true uplift.\n")
    out("| model | test WAPE | what-if uplift (mean) | recovery (what-if / true) | SHAP of test_promo (mean) | recovery (SHAP / expected SHAP) | rank of test_promo in SHAP |")
    out("|---|---|---|---|---|---|---|")
    stores = on["store_id"].astype(int).to_numpy()
    for c in "ABC":
        wape = evaluate(clean.assign(sales_true=clean.target), f"pred_{c}").set_index("group").loc["all", "wape"]
        wi = whatif_uplift(boosters[c], on, feats)
        sv, _ = shap_values(boosters[c], X, feats)
        rank = int(sv.abs().mean().rank(ascending=False)[PROMO_FEATURE])
        true_x = (X.demand_no_promo * effect).mean()
        rec = wi.mean() / true_uplift.mean()
        lo, hi = bootstrap_mean_ci(wi / true_uplift.mean(), stores)
        # SHAP is measured against the model's average forecast, which already contains promo days at rate `rate`,
        # so for an additive effect the expected SHAP on a promo row is (1 - rate) x true uplift.
        res[c] = {"whatif": wi.mean(), "rec": rec, "rec_lo": lo, "rec_hi": hi, "shap": sv[PROMO_FEATURE].mean(),
                  "shap_rec": sv[PROMO_FEATURE].mean() / (true_x * (1 - rate)), "rank": rank, "wape": wape}
        out(f"| {c} | {wape:.3f} | {wi.mean():+.4f} | {rec:.0%} ({lo:.0%}-{hi:.0%}) | {sv[PROMO_FEATURE].mean():+.4f} | "
            f"{res[c]['shap_rec']:.0%} | {rank} of {len(feats)} |")
    out("")
    check(0.8 <= res["A"]["rec"] <= 1.2, "A (clean) recovers the injected effect within +/-20% (the method works when data are clean)")
    check(res["B"]["rec"] < res["A"]["rec"], "B (censored) recovers less of the effect than A")
    out(f"- B under-recovers the true promo effect by {1 - res['B']['rec'] / res['A']['rec']:.0%} relative to A; "
        f"C differs from A by {res['C']['rec'] / res['A']['rec'] - 1:+.0%}.")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--effect", type=float, default=0.3)
    ap.add_argument("--rate", type=float, default=0.2)
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(MODELS / "lgb_config.json")
    out("# Step 8 report: faithfulness of SHAP explanations\n")
    curves = part1(a.rows)
    res = part2(a.effect, a.rate, cfg)
    pd.DataFrame([{"model": c, "deletion_gain": curves[c][1], "insertion_gain": curves[c][2],
                   "whatif_recovery": res[c]["rec"], "shap_recovery": res[c]["shap_rec"], "promo_rank": res[c]["rank"]}
                  for c in "ABC"]).to_csv(OUT / "step8_results.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    k = np.arange(1, len(FEATURES) + 1)
    for c, (cv, *_rest) in curves.items():
        ax[0].plot(k, cv["del_shap"], color=COLORS[c], label=f"{c} SHAP order")
        ax[0].plot(k, cv["del_random"], color=COLORS[c], ls=":", label=f"{c} random")
        ax[1].plot(k, cv["ins_shap"], color=COLORS[c], label=f"{c} SHAP order")
        ax[1].plot(k, cv["ins_random"], color=COLORS[c], ls=":", label=f"{c} random")
    ax[0].set(title="Deletion: share of forecast change after removing k features", xlabel="k features removed", ylabel="normalised change")
    ax[1].set(title="Insertion: share of forecast restored after adding k features", xlabel="k features restored")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "step8_curves.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    cs = list("ABC")
    ax.bar(cs, [res[c]["rec"] for c in cs], color=[COLORS[c] for c in cs],
           yerr=[[res[c]["rec"] - res[c]["rec_lo"] for c in cs], [res[c]["rec_hi"] - res[c]["rec"] for c in cs]], capsize=5)
    ax.axhline(1, color="k", ls="--", lw=1, label="true effect")
    ax.set(title="Recovery of a known +30% promo effect (what-if / truth)", ylabel="recovered share of true uplift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "step8_promo_recovery.png", dpi=110)
    plt.close(fig)
    out("\n![curves](step8_curves.png)\n\n![recovery](step8_promo_recovery.png)")

    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    report = OUT / "step8_faithfulness_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
