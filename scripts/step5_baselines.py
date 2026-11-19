"""Compute baseline forecasts and score them under the frn-50k-baseline evaluation protocol.

Outputs:
  outputs/step5_baselines.csv            metrics per baseline x condition x group
  outputs/step5_baselines_report.md
  data/processed/baseline_preds.parquet  test-day predictions for downstream comparison
Usage:
    .\\.venv\\Scripts\\python.exe scripts\\step5_baselines.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from blindspot.baselines import next_day_baselines, ssa_forecast
from blindspot.data import PROJECT_ROOT
from blindspot.metrics import bootstrap_wape, evaluate
from blindspot.prepare import PROCESSED_DIR, load_panel

lines: list[str] = []
problems: list[str] = []


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def check(ok: bool, msg: str) -> None:
    out(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        problems.append(msg)


def main() -> None:
    t0 = time.time()
    panel = load_panel(sample_only=True).sort_values(["series_id", "dt"]).reset_index(drop=True)
    cond = pd.read_parquet(PROCESSED_DIR / "conditions.parquet").sort_values(["series_id", "dt"]).reset_index(drop=True)
    stores = panel[["series_id", "store_id"]].drop_duplicates()

    # Next-day baselines rely on recorded test-time history, so they do not vary across conditions.
    fa = pd.read_parquet(PROCESSED_DIR / "features_A.parquet")
    nd = next_day_baselines(fa)
    test = nd[nd.is_test == 1].drop(columns=["store_id"]).merge(stores, on="series_id")

    # Week-ahead SSA uses each condition's 90-day training history separately.
    base = panel[["series_id", "dt", "discount", "holiday_flag", "precpt"]].copy()
    for c in "ABC":
        base[f"hist_{c}"] = cond[f"sales_{c}"].to_numpy()
        pred = ssa_forecast(base, f"hist_{c}")
        ev = base.index[(panel.split == "eval").to_numpy()]
        s = pd.Series(pred, index=ev)
        test = test.merge(pd.DataFrame({"series_id": panel.loc[ev, "series_id"].to_numpy(),
                                        "dt": panel.loc[ev, "dt"].to_numpy(), f"ssa_{c}": s.to_numpy()}),
                          on=["series_id", "dt"])
    ssa_raw = ssa_forecast(panel.assign(hist=panel.sale_amount), "hist")
    check(np.allclose(ssa_raw, test.sort_values(["series_id", "dt"]).ssa_A.to_numpy()),
          "SSA on condition A equals SSA on the raw dataset (A = recorded sales)")

    out("# Step 5 report: baselines\n")
    out(f"Scored on {int((test.is_clean_day == 1).sum()):,} clean test days of the 5,000-series sample. "
        "Metrics are averaged over the 7 test dates (frn-50k-baseline protocol).\n")
    rows = []
    specs = [("naive", "next-day", "all"), ("seasonal_naive", "next-day", "all"), ("moving_avg_7", "next-day", "all"),
             ("croston_sba", "next-day", "all")] + [(f"ssa_{c}", "week-ahead", c) for c in "ABC"]
    for col, horizon, condition in specs:
        m = evaluate(test, col)
        lo, hi = bootstrap_wape(test, col)
        m.insert(0, "condition", condition)
        m.insert(0, "horizon", horizon)
        m.insert(0, "model", col.replace("_A", "").replace("_B", "").replace("_C", ""))
        m["wape_ci95"] = np.where(m.group == "all", f"{lo:.3f}-{hi:.3f}", "")
        rows.append(m)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(PROJECT_ROOT / "outputs" / "step5_baselines.csv", index=False)

    out("## Overall (all series)\n")
    out("| model | horizon | trained on | WAPE (95% CI) | WPE (bias) | MAE | RMSE |")
    out("|---|---|---|---|---|---|---|")
    for r in res[res.group == "all"].itertuples():
        out(f"| {r.model} | {r.horizon} | {r.condition} | {r.wape:.3f} ({r.wape_ci95}) | {r.wpe:+.3f} | {r.mae:.3f} | {r.rmse:.3f} |")
    out("\n## By sales group (WAPE / WPE)\n")
    out("| model | trained on | low-sales | high-sales |")
    out("|---|---|---|---|")
    for (mdl, c), g in res.groupby(["model", "condition"], sort=False):
        lo_, hi_ = g[g.group == "low"].iloc[0], g[g.group == "high"].iloc[0]
        out(f"| {mdl} | {c} | {lo_.wape:.3f} / {lo_.wpe:+.3f} | {hi_.wape:.3f} / {hi_.wpe:+.3f} |")

    out("\n## Checks\n")
    check(test.groupby("series_id").size().eq(7).all(), "7 test days per series")
    check(int(test[["naive", "seasonal_naive", "moving_avg_7", "croston_sba", "ssa_A", "ssa_B", "ssa_C"]].isna().sum().sum()) == 0,
          "no missing baseline forecasts on test days")
    sa, sb = res.query("model=='ssa' and condition=='A' and group=='all'").iloc[0], res.query("model=='ssa' and condition=='B' and group=='all'").iloc[0]
    check(sb.wpe < sa.wpe, "SSA trained on censored history (B) is biased lower than on clean history (A)")

    keep = ["series_id", "dt", "store_id", "sales_group", "is_clean_day", "sales_true", "naive", "seasonal_naive",
            "moving_avg_7", "croston_sba", "ssa_A", "ssa_B", "ssa_C"]
    test[keep].to_parquet(PROCESSED_DIR / "baseline_preds.parquet", index=False)
    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    (PROJECT_ROOT / "outputs" / "step5_baselines_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
