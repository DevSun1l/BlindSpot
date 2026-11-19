"""Run promotion-lever what-if analyses for models A, B, and C.

Part 1: Real levers (`discount`, `activity_flag`) on clean test days, comparing
        B and C uplifts and discount curves against condition A.
Part 2: Synthetic injected promotion (`test_promo`) with known uplift, saving
        per-row errors for the downstream inventory simulation.

Outputs:
  outputs/step9_whatif_report.md, step9_discount_curve.png, step9_uplift.png
  data/processed/whatif_real.parquet, data/processed/whatif_injected.parquet
Usage:
    .\\.venv\\Scripts\\python.exe scripts\\step9_whatif.py [--effect 0.3] [--rate 0.2]
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
from blindspot.faithfulness import PROMO_FEATURE, train_injected  # noqa: E402
from blindspot.features import FEATURES  # noqa: E402
from blindspot.model import load_config, load_features, predict, split_rows  # noqa: E402
from blindspot.prepare import PROCESSED_DIR, load_panel  # noqa: E402
from blindspot.whatif import (DISCOUNT_GRID, SCENARIOS, apply_changes, bootstrap_ratio_ci, discount_curve,  # noqa: E402
                              observed_activity_ratio, whatif)

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


def guard_rail_checks() -> None:
    out("## Guard rails\n")
    X = pd.DataFrame({f: [1.0] for f in FEATURES})
    for bad, why in [({"lag_1": 5.0}, "changing past sales (lag_1)"), ({"roll_mean_7": 2.0}, "changing a rolling mean"),
                     ({"discount": 0.3}, "discount below training support (0.3)"),
                     ({"activity_flag": 0.5}, "non-binary activity_flag")]:
        try:
            apply_changes(X, bad)
            ok = False
        except ValueError:
            ok = True
        check(ok, f"what-if refuses {why}")
    out("")


def part1() -> tuple[dict, pd.DataFrame]:
    out("## Part 1: real levers (discount, activity_flag)\n")
    _, _, train_a, test = split_rows(load_features("A"))
    X = test[test.is_clean_day == 1].copy()
    stores = X["store_id"].astype(int).to_numpy()
    boosters = {c: lgb.Booster(model_file=str(MODELS / f"lgb_{c}.txt")) for c in "ABC"}

    ref = pd.read_parquet(PROCESSED_DIR / "lgb_preds.parquet")
    ref = X[["series_id", "dt"]].merge(ref, on=["series_id", "dt"], how="left")
    for c in "ABC":
        X[f"pred_{c}"] = predict(boosters[c], X)
    same = all(np.allclose(X[f"pred_{c}"].to_numpy(), ref[f"lgb_{c}"].to_numpy(), atol=1e-6) for c in "ABC")
    check(same, "forecasts with unchanged inputs equal Step 6's saved forecasts")

    out(f"\n{len(X):,} clean test days. Uplift = forecast(scenario) - forecast(baseline); relative uplift = "
        "mean uplift / mean baseline forecast. True effect is unknown for real levers, so B and C are compared with A. "
        "CI = 95% bootstrap over stores.\n")
    out("| scenario | model | mean baseline | mean uplift | relative uplift | ratio to A (95% CI) | rows with negative uplift |")
    out("|---|---|---|---|---|---|---|")
    res, keep = {}, X[["series_id", "dt", "store_id", "sales_group", "sales_true", "discount", "activity_flag"]
                      + [f"pred_{c}" for c in "ABC"]].copy()
    for name, (base, scen) in SCENARIOS.items():
        w = {c: whatif(boosters[c], X, base, scen) for c in "ABC"}
        for c in "ABC":
            u = w[c].uplift.to_numpy()
            lo, hi = bootstrap_ratio_ci(u, w["A"].uplift.to_numpy(), stores) if c != "A" else (1.0, 1.0)
            rel = u.mean() / w[c].base.mean()
            ratio = u.sum() / w["A"].uplift.sum()
            res[(name, c)] = {"uplift": u.mean(), "rel": rel, "ratio": ratio, "lo": lo, "hi": hi,
                              "neg": (u < -1e-9).mean()}
            ci = "-" if c == "A" else f"{ratio:.2f} ({lo:.2f}-{hi:.2f})"
            out(f"| {name} | {c} | {w[c].base.mean():.3f} | {u.mean():+.3f} | {rel:+.1%} | {ci} | {(u < -1e-9).mean():.1%} |")
        if name.startswith("promo"):
            for c in "ABC":
                keep[f"promo_base_{c}"], keep[f"promo_on_{c}"] = w[c].base.to_numpy(), w[c].scenario.to_numpy()
    out("")

    out("Relative uplift by sales group (scenario: promo = activity + 20% discount):\n")
    out("| group | A | B | C |")
    out("|---|---|---|---|")
    pb, pon = keep[[f"promo_base_{c}" for c in "ABC"]], keep[[f"promo_on_{c}" for c in "ABC"]]
    for grp in ["low", "high"]:
        m = (keep.sales_group == grp).to_numpy()
        vals = [(pon[f"promo_on_{c}"][m].mean() - pb[f"promo_base_{c}"][m].mean()) / pb[f"promo_base_{c}"][m].mean()
                for c in "ABC"]
        out(f"| {grp} | " + " | ".join(f"{v:+.1%}" for v in vals) + " |")
    out("")

    obs = observed_activity_ratio(train_a)
    out(f"- Descriptive reference (not causal): in condition A's clean training days, the median series sells "
        f"{obs - 1:+.0%} more on activity days than on non-activity days. Activity days also carry deeper discounts "
        "and other differences, so this is only a rough sanity check for the 'promo' scenario.")

    curves = {c: discount_curve(boosters[c], X) for c in "ABC"}
    out("\nDiscount curve (activity kept at each row's value), mean forecast relative to no discount:\n")
    out("| discount | " + " | ".join("ABC") + " |")
    out("|---|---|---|---|")
    for d in DISCOUNT_GRID:
        out(f"| {d:.2f} | " + " | ".join(f"{curves[c][d] / curves[c][1.0] - 1:+.1%}" for c in "ABC") + " |")
    out("")

    a_promo = res[("promo: activity + 20% discount", "A")]["rel"]
    check(a_promo > 0, "A forecasts more demand under a promo than without one (expected direction)")
    out(f"- A's promo uplift ({a_promo:+.1%}) is smaller than the descriptive gap ({obs - 1:+.0%}). Expected: the raw gap "
        "also contains whatever else differs on activity days (product mix, season, deeper discounts), which the "
        "model attributes to other features. The raw gap is not a target to match.")
    sparse = (train_a.discount < 0.6).mean()
    out(f"- Discounts below 0.60 are rare in training ({sparse:.1%} of rows), so the curve's last points rest on little "
        "data; the jump at 0.50 should not be trusted.")
    b = res[("promo: activity + 20% discount", "B")]
    out(f"- Promo scenario: B's uplift is {b['ratio']:.0%} of A's (CI {b['lo']:.0%}-{b['hi']:.0%}); "
        f"C's is {res[('promo: activity + 20% discount', 'C')]['ratio']:.0%}.")
    keep.to_parquet(PROCESSED_DIR / "whatif_real.parquet", index=False)
    return {"res": res, "curves": curves, "obs": obs}, keep


def part2(effect: float, rate: float, cfg: dict) -> dict:
    out(f"\n## Part 2: what-if error against a known effect (injected promo, +{effect:.0%} on {rate:.0%} of days)\n")
    boosters, test, _, _ = train_injected(load_panel(sample_only=True), cfg, rate=rate, effect=effect)
    feats = FEATURES + [PROMO_FEATURE]
    X = test[test.is_clean_day == 1].copy()
    X["true_uplift"] = (X.demand_no_promo * effect).astype(float)
    stores = X["store_id"].astype(int).to_numpy()
    out(f"All {len(X):,} clean test days: estimated uplift = forecast(test_promo=1) - forecast(test_promo=0); "
        f"true uplift = {effect:.0%} of the demand without promo. Error = estimated - true (negative = under-estimate).\n")
    out("| model | mean true uplift | mean estimated uplift | recovery (95% CI) | mean error | MAE of error | rows under-estimated |")
    out("|---|---|---|---|---|---|---|")
    keep = X[["series_id", "dt", "store_id", "sales_group", PROMO_FEATURE, "demand_no_promo", "sales_true",
              "true_uplift"]].copy()
    res = {}
    for c in "ABC":
        w = whatif(boosters[c], X, {PROMO_FEATURE: 0}, {PROMO_FEATURE: 1}, feats)
        err = w.uplift - X.true_uplift
        lo, hi = bootstrap_ratio_ci(w.uplift.to_numpy(), X.true_uplift.to_numpy(), stores)
        rec = w.uplift.sum() / X.true_uplift.sum()
        res[c] = {"rec": rec, "lo": lo, "hi": hi, "bias": err.mean(), "mae": err.abs().mean()}
        keep[f"base_{c}"], keep[f"promo_{c}"], keep[f"uplift_{c}"] = w.base, w.scenario, w.uplift
        out(f"| {c} | {X.true_uplift.mean():.4f} | {w.uplift.mean():.4f} | {rec:.0%} ({lo:.0%}-{hi:.0%}) | "
            f"{err.mean():+.4f} | {err.abs().mean():.4f} | {(err < 0).mean():.0%} |")
    out("\nRecovery by sales group:\n")
    out("| group | A | B | C |")
    out("|---|---|---|---|")
    for grp in ["low", "high"]:
        m = keep.sales_group == grp
        out(f"| {grp} | " + " | ".join(f"{keep.loc[m, f'uplift_{c}'].sum() / keep.loc[m, 'true_uplift'].sum():.0%}"
                                      for c in "ABC") + " |")
    out("")
    check(0.8 <= res["A"]["rec"] <= 1.2, "A recovers the known effect within +/-20% on all clean test days")
    check(res["B"]["hi"] < res["A"]["lo"], "B recovers significantly less than A (CIs do not overlap)")
    keep.to_parquet(PROCESSED_DIR / "whatif_injected.parquet", index=False)
    return res


def plots(p1: dict, p2: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in "ABC":
        cv = p1["curves"][c]
        ax.plot(cv.index, cv / cv[1.0] - 1, marker="o", color=COLORS[c], label=c)
    ax.invert_xaxis()
    ax.set(title="What-if: forecast change vs discount level (test week)", xlabel="discount (1.0 = none)",
           ylabel="mean forecast vs no discount")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "step9_discount_curve.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    names = list(SCENARIOS)
    x = np.arange(len(names))
    for i, c in enumerate("ABC"):
        ax[0].bar(x + (i - 1) * 0.25, [p1["res"][(n, c)]["rel"] for n in names], 0.25, color=COLORS[c], label=c)
    ax[0].set_xticks(x, [n.replace(" (", "\n(").replace(": ", ":\n") for n in names], fontsize=8)
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[0].set(title="Real levers: relative uplift by model", ylabel="mean uplift / mean baseline")
    ax[0].legend()
    cs = list("ABC")
    ax[1].bar(cs, [p2[c]["rec"] for c in cs], color=[COLORS[c] for c in cs],
              yerr=[[p2[c]["rec"] - p2[c]["lo"] for c in cs], [p2[c]["hi"] - p2[c]["rec"] for c in cs]], capsize=5)
    ax[1].axhline(1, color="k", ls="--", lw=1, label="true effect")
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[1].set(title="Known injected effect: what-if recovery", ylabel="estimated / true uplift")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "step9_uplift.png", dpi=110)
    plt.close(fig)
    out("\n![discount curve](step9_discount_curve.png)\n\n![uplift](step9_uplift.png)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--effect", type=float, default=0.3)
    ap.add_argument("--rate", type=float, default=0.2)
    a = ap.parse_args()
    t0 = time.time()
    out("# Step 9 report: what-if on the promotion levers\n")
    out("A what-if shows how a model's forecast responds to a planned change; it is not a causal estimate. "
        "Only discount and activity_flag (and the synthetic test_promo) can be changed; sales history cannot.\n")
    guard_rail_checks()
    p1, _ = part1()
    p2 = part2(a.effect, a.rate, load_config(MODELS / "lgb_config.json"))
    plots(p1, p2)
    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    report = OUT / "step9_whatif_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
