"""Step 6b: train LightGBM A, B and C with the frozen configuration and score them on the test week.

Each model is trained on all training targets (fit + validation days) of its condition, with the
same parameters, number of rounds and seed. Test inputs are identical for all three models.

Outputs:
  outputs/models/lgb_A.txt, lgb_B.txt, lgb_C.txt
  data/processed/lgb_preds.parquet
  outputs/step6_results.csv, outputs/step6_report.md
Usage:  .\\.venv\\Scripts\\python.exe scripts\\step6b_train.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from blindspot.data import PROJECT_ROOT
from blindspot.features import FEATURES
from blindspot.metrics import bootstrap_wape, evaluate
from blindspot.model import load_config, load_features, predict, split_rows, train_final
from blindspot.prepare import PROCESSED_DIR

MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
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
    cfg = load_config(MODELS_DIR / "lgb_config.json")
    base = pd.read_parquet(PROCESSED_DIR / "baseline_preds.parquet")

    preds, tests = {}, {}
    for c in "ABC":
        _, _, train, test = split_rows(load_features(c))
        booster = train_final(cfg["params"], cfg["n_rounds"], train)
        booster.save_model(str(MODELS_DIR / f"lgb_{c}.txt"))
        preds[c] = predict(booster, test)
        tests[c] = test
        print(f"trained {c}: {len(train):,} rows", flush=True)

    t = tests["A"][["series_id", "dt", "sales_group", "is_clean_day", "sales_true"]].copy()
    for c in "ABC":
        t[f"lgb_{c}"] = preds[c]
    t = t.merge(base[["series_id", "dt", "store_id", "moving_avg_7", "ssa_A", "ssa_B", "ssa_C"]], on=["series_id", "dt"])

    out("# Step 6 report: LightGBM A, B and C\n")
    p = cfg["params"]
    out(f"Frozen configuration (tuned on A): **{p['boosting']}**, objective {p['objective']}, "
        f"{cfg['n_rounds']} rounds, learning rate {p['learning_rate']}, {p['num_leaves']} leaves. "
        f"Same settings and seed for all three models.\n")

    out("## Checks\n")
    same = all(tests[c][FEATURES].equals(tests["A"][FEATURES]) for c in "BC")
    check(same, "test inputs are identical for A, B and C")
    check(all(np.isfinite(preds[c]).all() and (preds[c] >= 0).all() for c in "ABC"), "all test forecasts are finite and >= 0")

    rows = []
    for col, label in [("lgb_A", "LightGBM A (clean)"), ("lgb_B", "LightGBM B (censored)"), ("lgb_C", "LightGBM C (noise)"),
                       ("moving_avg_7", "Moving average 7 (best baseline)"), ("ssa_A", "SSA A"), ("ssa_B", "SSA B"), ("ssa_C", "SSA C")]:
        m = evaluate(t, col)
        lo, hi = bootstrap_wape(t, col)
        m.insert(0, "model", label)
        m["wape_lo"], m["wape_hi"] = np.where(m.group == "all", lo, np.nan), np.where(m.group == "all", hi, np.nan)
        rows.append(m)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(PROJECT_ROOT / "outputs" / "step6_results.csv", index=False)

    out("\n## Test-week results (clean test days, frn-50k-baseline protocol)\n")
    out("| model | WAPE (95% CI) | WPE (bias) | MAE | RMSE |")
    out("|---|---|---|---|---|")
    for r in res[res.group == "all"].itertuples():
        out(f"| {r.model} | {r.wape:.3f} ({r.wape_lo:.3f}-{r.wape_hi:.3f}) | {r.wpe:+.3f} | {r.mae:.3f} | {r.rmse:.3f} |")
    out("\n## By sales group (WAPE / WPE)\n")
    out("| model | low-sales | high-sales |")
    out("|---|---|---|")
    for mdl, g in res.groupby("model", sort=False):
        lo_, hi_ = g[g.group == "low"].iloc[0], g[g.group == "high"].iloc[0]
        out(f"| {mdl} | {lo_.wape:.3f} / {lo_.wpe:+.3f} | {hi_.wape:.3f} / {hi_.wpe:+.3f} |")

    a = res.query("model=='LightGBM A (clean)' and group=='all'").iloc[0]
    b = res.query("model=='LightGBM B (censored)' and group=='all'").iloc[0]
    c = res.query("model=='LightGBM C (noise)' and group=='all'").iloc[0]
    ma = res.query("model=='Moving average 7 (best baseline)' and group=='all'").iloc[0]
    out("\n## Comparison\n")
    out(f"- A vs best baseline: WAPE {a.wape:.3f} vs {ma.wape:.3f} ({a.wape / ma.wape - 1:+.1%})")
    out(f"- B vs A: WAPE {b.wape / a.wape - 1:+.1%}, bias {b.wpe - a.wpe:+.3f}")
    out(f"- C vs A: WAPE {c.wape / a.wape - 1:+.1%}, bias {c.wpe - a.wpe:+.3f}")
    clean = t[t.is_clean_day == 1]
    out(f"- Mean forecast on clean test days: A {clean.lgb_A.mean():.3f}, B {clean.lgb_B.mean():.3f}, "
        f"C {clean.lgb_C.mean():.3f}; true {clean.sales_true.mean():.3f}")
    out(f"- B forecasts lower than A on {(clean.lgb_B < clean.lgb_A).mean():.1%} of clean test days")
    check(a.wape < ma.wape, "LightGBM A beats the best baseline (moving average 7)")
    check(b.wpe < a.wpe, "B is biased lower than A")

    t.to_parquet(PROCESSED_DIR / "lgb_preds.parquet", index=False)
    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for pr in problems:
        out(f"- {pr}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    (PROJECT_ROOT / "outputs" / "step6_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
