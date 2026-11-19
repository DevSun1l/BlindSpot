"""Step 10: inventory simulation (newsvendor) for models A, B, C.

Part 1  Everyday ordering: order = forecast + safety allowance (CR-quantile of each model's own validation errors),
        scored against TRUE demand on the clean test days. Includes a sensitivity check over critical ratios and a
        diagnostic where B's safety allowance is calibrated on true (uncensored) errors.
Part 2  Promo decision (injected promo with a known +30% effect): a promo is planned for every clean test day;
        each model orders on its own what-if forecast. Compared with ordering on the TRUE effect (same base forecast,
        same safety), so the extra cost is caused only by the what-if error.

Outputs: outputs/step10_inventory_report.md, step10_results.csv, step10_costs.png, step10_cr_sensitivity.png
Usage:   .\\.venv\\Scripts\\python.exe scripts\\step10_inventory.py [--cu 1.0] [--co 0.5]
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blindspot.data import PROJECT_ROOT  # noqa: E402
from blindspot.explain import bootstrap_mean_ci  # noqa: E402
from blindspot.inventory import (CR_GRID, costs_for_cr, critical_ratio, order_quantity, outcomes,  # noqa: E402
                                 safety_quantiles, summarize)
from blindspot.model import load_config, load_features, predict, split_rows, train_final  # noqa: E402
from blindspot.prepare import PROCESSED_DIR  # noqa: E402

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


def validation_errors(cfg: dict) -> dict:
    """Per condition: validation rows with errors vs the condition's own target and vs true demand.
    Models are refit on the fit rows only (same frozen config), so validation errors are out-of-sample."""
    res = {}
    for c in "ABC":
        fit, valid, _, _ = split_rows(load_features(c))
        b = train_final(cfg["params"], cfg["n_rounds"], fit)
        p = predict(b, valid)
        res[c] = pd.DataFrame({"group": valid.sales_group.astype(str).to_numpy(),
                               "err_own": valid.target.to_numpy() - p, "err_true": valid.sales_true.to_numpy() - p,
                               "is_masked": valid.is_masked.to_numpy()})
    return res


def cost_ci(o_c: pd.DataFrame, o_a: pd.DataFrame, stores: np.ndarray) -> tuple[float, float]:
    """95% CI of (cost_c - cost_A) per unit of demand, bootstrapping stores."""
    scale = o_a["demand"].mean()
    return bootstrap_mean_ci(((o_c["cost"] - o_a["cost"]) / scale).to_numpy(), stores)


def part1(val: dict, cu: float, co: float) -> tuple[dict, pd.DataFrame]:
    cr = critical_ratio(cu, co)
    out(f"## Part 1: everyday ordering (critical ratio {cr:.2f}: lost sale = {cu:g}, wasted unit = {co:g})\n")
    t = pd.read_parquet(PROCESSED_DIR / "lgb_preds.parquet")
    t = t[t.is_clean_day == 1].reset_index(drop=True)
    g = t.sales_group.astype(str).to_numpy()
    D = t.sales_true.to_numpy()
    stores = t.store_id.astype(int).to_numpy()
    out(f"{len(t):,} clean test days scored against true demand. Safety = {cr:.0%} quantile of each model's validation "
        "errors (actual - forecast) against its own training target, per sales group.\n")

    out("| model | safety low | safety high | mean forecast / demand | fill rate | stockout days | short / demand | "
        "waste / demand | cost / demand | cost vs A (95% CI) |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    os_, res, safeties = {}, {}, {}
    variants = [(c, c, "err_own") for c in "ABC"] + [("B*", "B", "err_true")]
    for name, c, col in variants:
        sq = safety_quantiles(val[c][col].to_numpy(), val[c].group.to_numpy(), cr)
        safeties[name] = sq
        o = outcomes(order_quantity(t[f"lgb_{c}"].to_numpy(), g, sq), D, cu, co)
        os_[name] = o
        s = summarize(o)
        s["fc_ratio"] = t[f"lgb_{c}"].sum() / D.sum()
        res[name] = s
        if name == "A":
            ci = "-"
        else:
            lo, hi = cost_ci(o, os_["A"], stores)
            s["d_lo"], s["d_hi"] = lo, hi
            ci = f"{s['cost_per_demand'] / res['A']['cost_per_demand'] - 1:+.1%} (diff {lo:+.3f} to {hi:+.3f})"
        out(f"| {name} | {sq.get('low', np.nan):+.3f} | {sq.get('high', np.nan):+.3f} | {s['fc_ratio']:.3f} | "
            f"{s['fill_rate']:.1%} | {s['stockout_rate']:.1%} | {s['short_per_demand']:.3f} | {s['waste_per_demand']:.3f} | "
            f"{s['cost_per_demand']:.3f} | {ci} |")
    out("\nB* = model B, but its safety allowance is calibrated on TRUE validation errors (impossible in practice; "
        "diagnostic only). It separates the cost of B's low forecasts from the cost of B's mis-calibrated safety stock.\n")

    v = val["B"]
    m = v.is_masked == 1
    out(f"- B's validation errors: on masked days B's own target is censored, so the error looks "
        f"{v.loc[m, 'err_own'].mean():+.3f} on average instead of the true {v.loc[m, 'err_true'].mean():+.3f}; "
        "its planner therefore under-sizes the safety allowance.")

    by = []
    for grp in ["low", "high"]:
        mm = g == grp
        by.append(f"| {grp} | " + " | ".join(f"{os_[n].loc[mm, 'cost'].sum() / D[mm].sum():.3f}" for n in ["A", "B", "C"])
                  + " | " + " | ".join(f"{summarize(os_[n][mm])['fill_rate']:.1%}" for n in ["A", "B", "C"]) + " |")
    out("\nBy sales group:\n")
    out("| group | cost/demand A | B | C | fill rate A | B | C |")
    out("|---|---|---|---|---|---|---|")
    for row in by:
        out(row)

    out("\nSensitivity: cost / demand at other critical ratios (shortage cost fixed at 1, waste cost varied):\n")
    out("| critical ratio | waste cost | A | B | C | B vs A |")
    out("|---|---|---|---|---|---|")
    sens = {}
    for crx in CR_GRID:
        cux, cox = costs_for_cr(crx, cu)
        row = {}
        for c in "ABC":
            sq = safety_quantiles(val[c]["err_own"].to_numpy(), val[c].group.to_numpy(), crx)
            row[c] = summarize(outcomes(order_quantity(t[f"lgb_{c}"].to_numpy(), g, sq), D, cux, cox))
        sens[crx] = row
        out(f"| {crx:.2f} | {cox:.3f} | {row['A']['cost_per_demand']:.3f} | {row['B']['cost_per_demand']:.3f} | "
            f"{row['C']['cost_per_demand']:.3f} | {row['B']['cost_per_demand'] / row['A']['cost_per_demand'] - 1:+.1%} |")
    out("")
    check(res["B"]["d_lo"] > 0, "B's ordering costs significantly more than A's (CI above 0)")
    check(res["B"]["fill_rate"] < res["A"]["fill_rate"], "B fills less of the true demand than A")
    check(all(sens[x]["B"]["cost_per_demand"] > sens[x]["A"]["cost_per_demand"] for x in CR_GRID),
          "B costs more than A at every critical ratio tested")
    out(f"- B's WAPE was {0.314 / 0.286 - 1:+.0%} worse than A's (Step 6); its ordering cost is "
        f"{res['B']['cost_per_demand'] / res['A']['cost_per_demand'] - 1:+.0%} higher. "
        f"C (same-size noise) costs {res['C']['cost_per_demand'] / res['A']['cost_per_demand'] - 1:+.0%}.")
    return {"res": res, "sens": sens, "safeties": safeties}, t


def part2(safeties: dict, cu: float, co: float) -> dict:
    out("\n## Part 2: ordering for a planned promo (injected promo, known effect)\n")
    w = pd.read_parquet(PROCESSED_DIR / "whatif_injected.parquet")
    g = w.sales_group.astype(str).to_numpy()
    D = (w.demand_no_promo + w.true_uplift).to_numpy()
    stores = w.store_id.astype(int).to_numpy()
    pos = w.demand_no_promo > 0
    effect = float((w.true_uplift[pos] / w.demand_no_promo[pos]).median())
    out(f"A promo is planned for each of the {len(w):,} clean test days; true demand = demand without promo x "
        f"{1 + effect:.2f}. Each model orders: (its promo-off forecast + its what-if uplift) + safety. The reference "
        f"planner knows the TRUE effect (+{effect:.0%}) and orders the same promo-off forecast x {1 + effect:.2f} + the "
        "same safety, so any difference comes only from the what-if error (not from knowing the day's actual demand). "
        "Safety allowances are those from Part 1 (same condition, same rule).\n")
    out("| model | estimated / true uplift | order on what-if: fill rate | cost / demand | order on true effect: cost / demand | "
        "extra cost from what-if error (95% CI) | units under-ordered per 100 demand |")
    out("|---|---|---|---|---|---|---|")
    res = {}
    for c in "ABC":
        o_w = outcomes(order_quantity(w[f"promo_{c}"].to_numpy(), g, safeties[c]), D, cu, co)
        o_t = outcomes(order_quantity((w[f"base_{c}"] * (1 + effect)).to_numpy(), g, safeties[c]), D, cu, co)
        sw, st = summarize(o_w), summarize(o_t)
        lo, hi = bootstrap_mean_ci(((o_w.cost - o_t.cost) / D.mean()).to_numpy(), stores)
        under = (o_t.order.sum() - o_w.order.sum()) / D.sum() * 100
        res[c] = {"rec": w[f"uplift_{c}"].sum() / w.true_uplift.sum(), "cost_w": sw["cost_per_demand"],
                  "cost_t": st["cost_per_demand"], "extra": sw["cost_per_demand"] - st["cost_per_demand"],
                  "lo": lo, "hi": hi, "fill_w": sw["fill_rate"], "under": under}
        out(f"| {c} | {res[c]['rec']:.0%} | {sw['fill_rate']:.1%} | {sw['cost_per_demand']:.3f} | {st['cost_per_demand']:.3f} | "
            f"{res[c]['extra']:+.3f} ({lo:+.3f} to {hi:+.3f}) | {under:.1f} |")
    out("")
    check(res["B"]["lo"] > 0, "ordering on B's what-if costs significantly more than ordering on the true effect")
    check(res["B"]["extra"] > res["A"]["extra"], "B's what-if error costs more than A's")
    out(f"- Ordering on B's what-if adds {res['B']['extra'] / res['B']['cost_t']:+.0%} to the cost of a promo day versus "
        f"knowing the true effect; for A the figure is {res['A']['extra'] / res['A']['cost_t']:+.0%}, for C "
        f"{res['C']['extra'] / res['C']['cost_t']:+.0%}.")
    return res


def plots(p1: dict, p2: dict, cu: float, co: float) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    names = ["A", "B", "C", "B*"]
    short = [p1["res"][n]["short_per_demand"] * cu for n in names]
    waste = [p1["res"][n]["waste_per_demand"] * co for n in names]
    cols = [COLORS.get(n, "#B0B0B0") for n in names]
    ax[0].bar(names, short, color=cols, label="lost-sale cost")
    ax[0].bar(names, waste, bottom=short, color=cols, alpha=0.45, hatch="//", label="waste cost")
    ax[0].set(title="Everyday ordering: cost per unit of true demand", ylabel="cost / demand",
              ylim=(0, max(s + w for s, w in zip(short, waste)) * 1.3))
    ax[0].legend(loc="upper left", ncol=2)
    cs = list("ABC")
    x = np.arange(3)
    ax[1].bar(x - 0.2, [p2[c]["cost_t"] for c in cs], 0.4, color="#B0B0B0", label="order on true effect")
    ax[1].bar(x + 0.2, [p2[c]["cost_w"] for c in cs], 0.4, color=[COLORS[c] for c in cs], label="order on model's what-if")
    ax[1].set_xticks(x, cs)
    ax[1].set(title="Planned promo day: cost per unit of true demand", ylabel="cost / demand",
              ylim=(0, max(p2[c]["cost_w"] for c in cs) * 1.3))
    ax[1].legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "step10_costs.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for c in "ABC":
        ax.plot(CR_GRID, [p1["sens"][x][c]["cost_per_demand"] for x in CR_GRID], marker="o", color=COLORS[c], label=c)
    ax.set(title="Ordering cost vs critical ratio (lost-sale cost fixed at 1)", xlabel="critical ratio cu/(cu+co)",
           ylabel="cost / demand")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "step10_cr_sensitivity.png", dpi=110)
    plt.close(fig)
    out("\n![costs](step10_costs.png)\n\n![sensitivity](step10_cr_sensitivity.png)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cu", type=float, default=1.0, help="cost of one lost sale (relative)")
    ap.add_argument("--co", type=float, default=0.5, help="cost of one wasted unit (relative)")
    a = ap.parse_args()
    t0 = time.time()
    out("# Step 10 report: inventory simulation\n")
    out("Assumptions (no prices, costs or lead times in the data): perishable products ordered daily for next-day "
        "delivery; unsold units are wasted, unmet demand is lost; costs are relative. Sales units are normalised, so "
        "costs are reported per unit of true demand. Same ordering rule for every model.\n")
    val = validation_errors(load_config(MODELS / "lgb_config.json"))
    p1, _ = part1(val, a.cu, a.co)
    p2 = part2(p1["safeties"], a.cu, a.co)
    plots(p1, p2, a.cu, a.co)

    rows = [{"part": "everyday", "model": n, **{k: v for k, v in s.items() if not k.startswith("d_")}}
            for n, s in p1["res"].items()]
    rows += [{"part": "promo", "model": c, **s} for c, s in p2.items()]
    pd.DataFrame(rows).to_csv(OUT / "step10_results.csv", index=False)

    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    report = OUT / "step10_inventory_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
