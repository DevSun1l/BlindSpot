"""Step 4: build leakage-safe feature tables for conditions A, B and C.

Outputs:
  data/processed/features_A.parquet, features_B.parquet, features_C.parquet
  outputs/step4_features_report.md
Usage:  .\\.venv\\Scripts\\python.exe scripts\\step4_build_features.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from blindspot.data import PROJECT_ROOT
from blindspot.features import FEATURE_GROUPS, FEATURES, FORBIDDEN, LAGS, VALID_DAYS, WARMUP_DAYS, build_features
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
    panel = load_panel(sample_only=True)
    cond = pd.read_parquet(PROCESSED_DIR / "conditions.parquet")
    F = {c: build_features(cond, panel, c) for c in "ABC"}

    out("# Step 4 report: features\n")
    out("## Feature list\n")
    out("| group | features |")
    out("|---|---|")
    for g, fs in FEATURE_GROUPS.items():
        out(f"| {g} | {', '.join(fs)} |")
    out(f"\n{len(FEATURES)} features. Next-day forecast; sales history uses days up to t-1 only.")

    a = F["A"]
    out("\n## Rows\n")
    out(f"- Train targets (clean train days, day >= {WARMUP_DAYS}): {a.is_train_target.sum():,} "
        f"(of which validation, last {VALID_DAYS} train days: {a.is_valid.sum():,})")
    out(f"- Test rows: {a.is_test.sum():,}; scored (clean test days): {(a.is_test.astype(bool) & (a.is_clean_day == 1)).sum():,}")
    out(f"- Masked days among train targets: {a[a.is_train_target == 1].is_masked.mean():.1%}")

    out("\n## Leakage and consistency checks\n")
    check(not FORBIDDEN & set(FEATURES), "no stockout or masking column is a feature")
    for c, f in F.items():
        prev = cond.sort_values(["series_id", "dt"]).groupby("series_id")[f"sales_{c}"].shift(1).to_numpy()
        tr = (f.split == "train").to_numpy() & ~np.isnan(prev)
        check(np.allclose(f.lag_1.to_numpy()[tr], prev[tr]), f"{c}: lag_1 on train rows = previous day's condition-{c} sales")
        pm = f.groupby("series_id").target.shift(1)
        manual = pm.groupby(f.series_id).transform(lambda s: s.rolling(7).mean())
        ok = (f.split == "train") & manual.notna()
        check(np.allclose(f.roll_mean_7[ok], manual[ok], atol=1e-5), f"{c}: roll_mean_7 = mean of the 7 previous days (excludes day t)")
    te = a.is_test == 1
    same = all(np.allclose(F[c].loc[te, FEATURE_GROUPS["recent_sales"]].to_numpy(dtype=float),
                           a.loc[te, FEATURE_GROUPS["recent_sales"]].to_numpy(dtype=float), equal_nan=True) for c in "BC")
    check(same, "test rows have identical features in A, B and C")
    check(all((F[c].loc[te, "target"] == a.loc[te, "target"]).all() for c in "BC"), "test targets are the recorded (true) sales in all conditions")
    tt = a.is_train_target == 1
    check(int(a.loc[tt, FEATURES[:len(LAGS) + 3]].isna().sum().sum()) == 0, "no missing history features on train targets")
    trn = (a.split == "train") & a.lag_1.notna()
    check(bool((F["B"].loc[trn, "lag_1"] <= a.loc[trn, "lag_1"] + 1e-6).all()), "B's lags never exceed A's (censoring flows into history)")
    out(f"- [INFO] B's lag_1 is lower than A's on {(F['B'].loc[trn, 'lag_1'] < a.loc[trn, 'lag_1'] - 1e-6).mean():.1%} of train rows")
    check(a.loc[a.is_valid == 1, "day_idx"].min() > a.loc[(a.is_train_target == 1) & (a.is_valid == 0), "day_idx"].max(),
          "validation days come strictly after the fitting days")

    out("\n## How censoring changes the training features (train targets)\n")
    out("| feature | A mean | B mean | C mean | B vs A |")
    out("|---|---|---|---|---|")
    for col in ["target", "lag_1", "lag_7", "roll_mean_7", "roll_mean_14", "roll_std_7"]:
        m = {c: F[c].loc[F[c].is_train_target == 1, col].mean() for c in "ABC"}
        out(f"| {col} | {m['A']:.3f} | {m['B']:.3f} | {m['C']:.3f} | {m['B'] / m['A'] - 1:+.1%} |")
    out("\nCorrelation of each sales feature with the target (train targets):\n")
    out("| feature | A | B | C |")
    out("|---|---|---|---|")
    for col in FEATURE_GROUPS["recent_sales"]:
        r = {c: F[c].loc[F[c].is_train_target == 1, [col, "target"]].corr().iloc[0, 1] for c in "ABC"}
        out(f"| {col} | {r['A']:.3f} | {r['B']:.3f} | {r['C']:.3f} |")

    for c, f in F.items():
        f.to_parquet(PROCESSED_DIR / f"features_{c}.parquet", index=False)
    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    (PROJECT_ROOT / "outputs" / "step4_features_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
