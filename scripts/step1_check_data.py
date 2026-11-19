"""Download FreshRetailNet-50K and validate schema, completeness, and stockout encoding.

Writes a validation report to `outputs/step1_data_report.md`.
Usage:
    .\\.venv\\Scripts\\python.exe scripts\\step1_check_data.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from blindspot.data import (BUSINESS_HOURS, EVAL_DAYS, EXPECTED_COLUMNS, KEYS, OUT_OF_STOCK,
                            PROJECT_ROOT, TRAIN_DAYS, load_split)

lines: list[str] = []
problems: list[str] = []


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def check(ok: bool, msg: str) -> None:
    out(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        problems.append(msg)


def hourly_matrix(series: pd.Series) -> np.ndarray:
    return np.stack(series.to_numpy())


def check_split(name: str, df: pd.DataFrame, days: int) -> None:
    out(f"\n## Split: {name}\n")
    n_series = df.groupby(["store_id", "product_id"]).ngroups
    out(f"- Rows: {len(df):,}  | series (store x product): {n_series:,}")
    out(f"- Stores: {df.store_id.nunique():,} | products: {df.product_id.nunique():,} | "
        f"cities: {df.city_id.nunique():,}")
    out(f"- Date range: {df.dt.min().date()} -> {df.dt.max().date()} ({df.dt.nunique()} days)")

    check(list(df.columns) == EXPECTED_COLUMNS or set(df.columns) == set(EXPECTED_COLUMNS),
          f"19 expected columns present (got {len(df.columns)})")
    check(df.dt.nunique() == days, f"{days} distinct dates")
    per_series = df.groupby(["store_id", "product_id"]).size()
    check((per_series == days).all(), f"every series has exactly {days} rows "
          f"(min={per_series.min()}, max={per_series.max()})")
    check(not df.duplicated(KEYS).any(), "no duplicate (store_id, product_id, dt) keys")

    scalar_cols = [c for c in df.columns if c not in ("hours_sale", "hours_stock_status")]
    na = df[scalar_cols].isna().sum()
    na = na[na > 0]
    check(na.empty, "no missing values in scalar columns" + ("" if na.empty else f": {na.to_dict()}"))
    check((df.sale_amount >= 0).all(), "sale_amount is never negative")

    out("\n**Column summary**\n")
    out("| column | dtype | min | max | mean |")
    out("|---|---|---|---|---|")
    for c in scalar_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            out(f"| {c} | {df[c].dtype} | {df[c].min():.4g} | {df[c].max():.4g} | {df[c].mean():.4g} |")
        else:
            out(f"| {c} | {df[c].dtype} | {df[c].min()} | {df[c].max()} | - |")

    # Validate hourly sales and stockout status arrays.
    hs = hourly_matrix(df["hours_sale"])
    st = hourly_matrix(df["hours_stock_status"])
    check(hs.shape[1] == 24 and st.shape[1] == 24, f"hourly arrays have 24 values (got {hs.shape[1]}, {st.shape[1]})")
    vals = np.unique(st)
    check(set(vals.tolist()) <= {0, 1}, f"hours_stock_status values are 0/1 (found {vals.tolist()})")

    # Confirm OUT_OF_STOCK count over business hours (6..21) matches stock_hour6_22_cnt.
    oos_count = (st[:, BUSINESS_HOURS] == OUT_OF_STOCK).sum(axis=1)
    match = (oos_count == df.stock_hour6_22_cnt.to_numpy()).mean()
    alt = ((st[:, BUSINESS_HOURS] == 1 - OUT_OF_STOCK).sum(axis=1) == df.stock_hour6_22_cnt.to_numpy()).mean()
    check(match > 0.99, f"stockout encoding: value {OUT_OF_STOCK} = out of stock "
          f"(matches stock_hour6_22_cnt on {match:.2%} of rows; opposite encoding matches {alt:.2%})")

    # Check consistency between daily totals and sum of hourly sales.
    hsum = hs.sum(axis=1)
    rel = np.abs(hsum - df.sale_amount.to_numpy()) / np.maximum(df.sale_amount.to_numpy(), 1e-9)
    close = np.isclose(hsum, df.sale_amount.to_numpy(), rtol=1e-3, atol=1e-6).mean()
    out(f"- [INFO] sum(hours_sale) equals sale_amount on {close:.2%} of rows "
        f"(median relative gap {np.median(rel):.3g})")
    biz_share = hs[:, BUSINESS_HOURS].sum() / max(hs.sum(), 1e-9)
    out(f"- [INFO] share of sales inside business hours 6-22: {biz_share:.2%}")

    # Stockout prevalence and start-hour distribution.
    so = df.stock_hour6_22_cnt > 0
    out(f"\n**Stockouts**\n")
    out(f"- Rows with >=1 stockout hour: {so.mean():.2%} (paper reports ~44%)")
    out(f"- Rows fully out of stock (16 h): {(df.stock_hour6_22_cnt >= 16).mean():.2%}")
    clean_per_series = (~so).groupby([df.store_id, df.product_id]).sum()
    out(f"- Stockout-free (clean) days per series: median {clean_per_series.median():.0f}, "
        f"min {clean_per_series.min()}, max {clean_per_series.max()}")
    out(f"- Series with >= 30 clean days: {(clean_per_series >= 30).mean():.2%}; "
        f"with 0 clean days: {(clean_per_series == 0).mean():.2%}")
    out(f"- Zero-sale rows: {(df.sale_amount == 0).mean():.2%} "
        f"(of which on stockout days: {(so & (df.sale_amount == 0)).sum() / max((df.sale_amount == 0).sum(), 1):.2%})")
    first_oos_hour = np.where(st[:, BUSINESS_HOURS].any(axis=1),
                              st[:, BUSINESS_HOURS].argmax(axis=1) + 6, -1)
    fo = pd.Series(first_oos_hour[first_oos_hour >= 0]).value_counts().sort_index()
    out(f"- First stockout hour distribution (hour: share of stockout days): "
        + ", ".join(f"{h}:{v / fo.sum():.0%}" for h, v in fo.items()))

    out(f"\n**Promotions**\n")
    out(f"- discount < 1 on {(df.discount < 1).mean():.2%} of rows; activity_flag=1 on "
        f"{(df.activity_flag == 1).mean():.2%}; holiday_flag=1 on {(df.holiday_flag == 1).mean():.2%}")


def main() -> None:
    t0 = time.time()
    out("# Step 1 data report: FreshRetailNet-50K\n")
    train = load_split("train")
    evl = load_split("eval")
    out(f"Loaded in {time.time() - t0:.0f}s (cached at data/raw/).")
    check_split("train", train, TRAIN_DAYS)
    check_split("eval", evl, EVAL_DAYS)

    out("\n## Train/eval consistency\n")
    tk = set(map(tuple, train[["store_id", "product_id"]].drop_duplicates().to_numpy()))
    ek = set(map(tuple, evl[["store_id", "product_id"]].drop_duplicates().to_numpy()))
    check(tk == ek, f"same series in train and eval ({len(tk):,} vs {len(ek):,})")
    check(train.dt.max() < evl.dt.min(), f"eval starts after train ends "
          f"({train.dt.max().date()} < {evl.dt.min().date()})")

    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    report = PROJECT_ROOT / "outputs" / "step1_data_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()
