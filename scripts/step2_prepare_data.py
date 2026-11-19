"""Clean and assemble the train+eval panel, draw the stratified study sample, and write a report.

Outputs:
  data/processed/panel.parquet          cleaned panel for all 50,000 series
  data/processed/series.parquet         per-series metadata and eligibility flags
  data/processed/sample_series.parquet  stratified study sample (seed 42)
  outputs/step2_prepare_report.md
Usage:
    .\\.venv\\Scripts\\python.exe scripts\\step2_prepare_data.py [--sample-size 5000]
"""
from __future__ import annotations

import argparse
import time

from blindspot.data import PROJECT_ROOT, load_split
from blindspot.prepare import (MIN_CLEAN_EVAL_DAYS, MIN_CLEAN_TRAIN_DAYS, PROCESSED_DIR, SEED,
                               DEFAULT_SAMPLE_SIZE, build_panel, draw_sample, series_table)

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    args = ap.parse_args()
    t0 = time.time()

    out("# Step 2 report: cleaning and preparation\n")
    panel = build_panel()
    raw_rows = len(load_split("train")) + len(load_split("eval"))

    out("## Cleaning\n")
    check(len(panel) == raw_rows, f"no rows lost ({len(panel):,} rows = train + eval)")
    fixed = panel.discount_fixed.sum()
    out(f"- discount fixed on {fixed:,} rows ({fixed / len(panel):.3%}): <=0 -> 1.0, >1 -> 1.0")
    check(panel.discount.between(0, 1, inclusive="right").all(), "discount now within (0, 1]")
    check((panel.is_clean_day + panel.is_stockout_day == 1).all(), "every day is clean XOR stockout")
    check(panel.groupby("series_id").size().eq(97).all(), "every series has 97 days (90 train + 7 eval)")
    check(panel.groupby("series_id").day_idx.apply(lambda s: s.is_monotonic_increasing and s.diff().dropna().eq(1).all()).all(),
          "days are consecutive with no gaps in every series")
    check(not ({"stock_hour6_22_cnt"} & set(panel.columns)), "raw stockout count renamed to stockout_hours")

    tab = series_table(panel)
    out("\n## Series eligibility\n")
    out(f"- Rules: >= {MIN_CLEAN_TRAIN_DAYS} clean train days, >= {MIN_CLEAN_EVAL_DAYS} clean eval day, mean train sales > 0")
    out(f"- Eligible: {tab.eligible.sum():,} of {len(tab):,} series ({tab.eligible.mean():.1%})")
    out(f"- Sales groups (all series): low {(tab.sales_group == 'low').sum():,}, high {(tab.sales_group == 'high').sum():,}")

    sample = draw_sample(tab, n=args.sample_size, seed=SEED)
    out(f"\n## Study sample (n={len(sample):,}, seed={SEED}, stratified by first category x sales group)\n")
    check(len(sample) == min(args.sample_size, tab.eligible.sum()), "sample has the requested size")
    check(sample.series_id.is_unique and sample.eligible.all(), "sample is unique and all eligible")
    elig = tab[tab.eligible]
    out("")
    out("| measure | eligible pool | sample |")
    out("|---|---|---|")
    for name, f in [
        ("share low-sales series", lambda d: f"{(d.sales_group == 'low').mean():.1%}"),
        ("mean train sales", lambda d: f"{d.train_mean_sale.mean():.3f}"),
        ("median clean train days", lambda d: f"{d.clean_train_days.median():.0f}"),
        ("mean stockout rate (train)", lambda d: f"{d.stockout_rate_train.mean():.1%}"),
        ("mean promo rate (train)", lambda d: f"{d.promo_rate_train.mean():.1%}"),
        ("stores / products / categories", lambda d: f"{d.store_id.nunique()} / {d.product_id.nunique()} / {d.first_category_id.nunique()}"),
    ]:
        out(f"| {name} | {f(elig)} | {f(sample)} |")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED_DIR / "panel.parquet", index=False)
    tab.to_parquet(PROCESSED_DIR / "series.parquet", index=False)
    sample.to_parquet(PROCESSED_DIR / "sample_series.parquet", index=False)

    s = panel[panel.series_id == sample.series_id.iloc[0]]
    out(f"\n## Example series (series_id={s.series_id.iloc[0]}, store {s.store_id.iloc[0]}, product {s.product_id.iloc[0]}), last 10 days\n")
    out("| dt | split | sale_amount | stockout_hours | clean | discount | activity |")
    out("|---|---|---|---|---|---|---|")
    for r in s.tail(10).itertuples():
        out(f"| {r.dt.date()} | {r.split} | {r.sale_amount:.2f} | {r.stockout_hours} | {r.is_clean_day} | {r.discount:.2f} | {r.activity_flag} |")

    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    (PROJECT_ROOT / "outputs" / "step2_prepare_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
