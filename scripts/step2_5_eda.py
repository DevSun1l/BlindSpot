"""Exploratory data analysis on the 5,000-series study sample.

Restricts all plots and summaries to training days, except the descriptive
train-vs-test comparison at the end. "Relative sales" normalizes each day's
sales by the series' mean on stockout-free training days.

Outputs: outputs/eda/*.png and outputs/eda/eda_summary.md
Usage:
    .\\.venv\\Scripts\\python.exe scripts\\step2_5_eda.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blindspot.data import BUSINESS_HOURS, PROJECT_ROOT  # noqa: E402
from blindspot.prepare import load_panel  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

lines: list[str] = []


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    out(f"\n![{name}]({name})")


def main() -> None:
    p = load_panel(sample_only=True)
    tr = p[p.split == "train"].copy()
    clean_mean = tr[tr.is_clean_day == 1].groupby("series_id").sale_amount.mean().rename("clean_mean")
    p = p.merge(clean_mean, on="series_id")
    tr = tr.merge(clean_mean, on="series_id")
    tr["rel"] = tr.sale_amount / tr.clean_mean
    tr["weekday"] = tr.dt.dt.dayofweek
    clean = tr[tr.is_clean_day == 1]

    out("# Step 2.5 EDA summary (study sample: 5,000 series, train days unless stated)\n")

    # 1. Sales distribution and intermittency
    out("## 1. Sales distribution and intermittency")
    zero = (tr.sale_amount == 0).mean()
    nz = tr[tr.sale_amount > 0]

    def sb_class(s: pd.Series) -> str:
        """Syntetos-Boylan class from ADI (avg interval between non-zero days) and CV^2 of non-zero sizes."""
        nzv = s[s > 0]
        adi = len(s) / max(len(nzv), 1)
        cv2 = (nzv.std() / nzv.mean()) ** 2 if len(nzv) > 1 else 0.0
        return ("intermittent" if cv2 < 0.49 else "lumpy") if adi >= 1.32 else ("smooth" if cv2 < 0.49 else "erratic")

    cls = clean.groupby("series_id").sale_amount.apply(sb_class).value_counts(normalize=True)
    out(f"- Zero-sale days: {zero:.1%} of all train days; {(clean.sale_amount == 0).mean():.1%} of clean days")
    out(f"- Daily sales quantiles (50/90/99%): {tr.sale_amount.quantile([.5, .9, .99]).round(2).tolist()}")
    out("- Syntetos-Boylan demand class (clean days): " + ", ".join(f"{k} {v:.1%}" for k, v in cls.items()))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(np.log1p(tr.sale_amount), bins=80, color="#4C78A8")
    ax[0].set(title="Daily sales (log1p scale)", xlabel="log(1 + sale_amount)", ylabel="days")
    for g, c in [("low", "#F58518"), ("high", "#4C78A8")]:
        ax[1].hist(tr[tr.sales_group == g].sale_amount.clip(upper=10), bins=50, alpha=.6, label=g, color=c, density=True)
    ax[1].set(title="Daily sales by sales group (clipped at 10)", xlabel="sale_amount", ylabel="density")
    ax[1].legend()
    ax[2].bar(cls.index, cls.values, color="#54A24B")
    ax[2].set(title="Demand class (Syntetos-Boylan)", ylabel="share of series")
    save(fig, "01_sales_distribution.png")
    out("- Implication: demand is mostly smooth/erratic at the daily level, not truly intermittent; "
        "Croston-SBA is kept as a baseline but is not expected to win.")

    # 2. Example series
    out("\n## 2. Example series with stockout days")
    rng = np.random.default_rng(7)
    ids = list(rng.choice(p[p.sales_group == "high"].series_id.unique(), 3, replace=False)) + \
        list(rng.choice(p[p.sales_group == "low"].series_id.unique(), 3, replace=False))
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
    for ax, sid in zip(axes.T.ravel(), ids):
        s = p[p.series_id == sid]
        ax.bar(s.dt, s.stockout_hours / 16 * s.sale_amount.max(), color="#E45756", alpha=.25, width=1,
               label="stockout hours (scaled)")
        ax.plot(s.dt, s.sale_amount, color="#4C78A8", lw=1.2, label="sales")
        ax.axvline(pd.Timestamp("2024-06-26"), color="k", ls="--", lw=.8)
        ax.set_title(f"series {sid} (store {s.store_id.iloc[0]}, product {s.product_id.iloc[0]}, "
                     f"{s.sales_group.iloc[0]})", fontsize=9)
    axes[0, 0].legend(fontsize=8)
    save(fig, "02_example_series.png")
    out("- Red bars = stockout hours that day (taller = longer stockout); dashed line = start of test week.")

    # 3. Sales on stockout vs clean days
    out("\n## 3. How much do stockouts hide?")
    bins = pd.cut(tr.stockout_hours, [-1, 0, 4, 8, 12, 15, 16], labels=["0", "1-4", "5-8", "9-12", "13-15", "16"])
    rel_by = tr.groupby(bins, observed=True).rel.mean()
    for k, v in rel_by.items():
        out(f"- stockout hours {k}: sales = {v:.2f} x the series' clean-day mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rel_by.index.astype(str), rel_by.values, color="#E45756")
    ax.axhline(1, color="k", lw=.8)
    ax.set(title="Relative sales by stockout hours that day", xlabel="stockout hours (6-22)", ylabel="sales / clean-day mean")
    save(fig, "03_sales_vs_stockout_hours.png")
    out("- Note: this mixes lost sales with selection (items that sell out may have had high demand). "
        "It shows observed sales are depressed, not the true size of hidden demand.")

    # 4. Hourly profile and stockout start hour
    out("\n## 4. Hourly sales profile and stockout start hour")
    hs = np.stack(clean.hours_sale.to_numpy())
    share = hs.sum(axis=0) / hs.sum()
    after = 1 - np.cumsum(share) + share  # share of daily sales from hour h onward
    st = np.stack(tr.hours_stock_status.to_numpy())[:, BUSINESS_HOURS]
    has = st.any(axis=1)
    first = st[has].argmax(axis=1) + 6
    fh = pd.Series(first).value_counts(normalize=True).sort_index()
    expected_loss = float((fh * pd.Series(after[fh.index], index=fh.index)).sum())
    out(f"- Peak sales hours (clean days): {', '.join(f'{h}:00' for h in np.argsort(share)[::-1][:3])}")
    out(f"- If a stockout starts at the observed start-hour mix and lasts the rest of the day, the expected "
        f"hidden share of daily sales is about {expected_loss:.0%}. This is the scale of censoring Step 3 will impose.")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].bar(range(24), share, color="#4C78A8", label="share of daily sales")
    ax[0].plot(range(24), after, color="#E45756", marker=".", label="share sold from this hour on")
    ax[0].axvspan(5.5, 21.5, color="grey", alpha=.08)
    ax[0].set(title="Hourly sales profile (clean days)", xlabel="hour", xticks=range(0, 24, 2))
    ax[0].legend()
    ax[1].bar(fh.index, fh.values, color="#E45756")
    ax[1].set(title="Hour the stockout starts (stockout days)", xlabel="hour", ylabel="share", xticks=range(6, 22))
    save(fig, "04_hourly_profile_stockout_start.png")

    # 5. Weekday and holiday effects
    out("\n## 5. Weekday and holiday effects (clean days)")
    wd = clean.groupby("weekday").rel.mean()
    hol = clean.groupby("holiday_flag").rel.mean()
    out("- Relative sales by weekday: " + ", ".join(f"{WEEKDAYS[i]} {v:.2f}" for i, v in wd.items()))
    out(f"- Holiday {hol.get(1, np.nan):.2f} vs non-holiday {hol.get(0, np.nan):.2f}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar([WEEKDAYS[i] for i in wd.index], wd.values, color="#4C78A8")
    ax[0].set(title="Relative sales by weekday", ylim=(wd.min() * .9, wd.max() * 1.05))
    ax[1].bar(["non-holiday", "holiday"], [hol.get(0), hol.get(1)], color="#72B7B2")
    ax[1].set(title="Relative sales: holiday flag")
    save(fig, "05_weekday_holiday.png")
    out(f"- Note: holiday_flag is 1 on {tr.holiday_flag.mean():.0%} of days, so it likely includes weekends; "
        f"compare with the weekday chart.")

    # 6. Promotion effect
    out("\n## 6. Promotion effect (the what-if levers, clean days)")
    db = pd.cut(clean.discount, [0, .6, .7, .8, .9, .99, 1.0], labels=["<=0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.99", "1.0 (none)"])
    disc = clean.groupby(db, observed=True).rel.agg(["mean", "size"])
    act = clean.groupby("activity_flag").rel.mean()
    for k, r in disc.iterrows():
        out(f"- discount {k}: relative sales {r['mean']:.2f} ({int(r['size']):,} days)")
    out(f"- activity_flag 1 vs 0: {act.get(1):.2f} vs {act.get(0):.2f} (uplift {act.get(1) / act.get(0) - 1:+.0%})")
    within = clean.groupby(["series_id", "activity_flag"]).rel.mean().unstack().dropna()
    out(f"- Within the same series (series with both promo and non-promo clean days, n={len(within):,}): "
        f"median uplift {(within[1] / within[0]).median() - 1:+.0%}")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(disc.index.astype(str), disc["mean"], color="#F58518")
    ax[0].set(title="Relative sales by discount level", xlabel="discount (1.0 = full price)")
    ax[1].bar(["no activity", "activity"], [act.get(0), act.get(1)], color="#F58518")
    ax[1].set(title="Relative sales by activity_flag")
    save(fig, "06_promotion_effect.png")
    out("- Caveat: discounts are not randomly assigned, so these are associations, not causal uplifts.")

    # 7. Autocorrelation
    out("\n## 7. Autocorrelation of daily sales (lags 1-14)")
    lags = range(1, 15)
    acf = {}
    for lag in lags:
        acf[lag] = tr.groupby("series_id").sale_amount.apply(lambda s: s.autocorr(lag)).median()
    acf = pd.Series(acf)
    out("- Median per-series autocorrelation: " + ", ".join(f"lag{k} {v:.2f}" for k, v in acf.items()))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(acf.index, acf.values, color="#4C78A8")
    ax.set(title="Median autocorrelation of daily sales", xlabel="lag (days)", xticks=list(lags))
    save(fig, "07_autocorrelation.png")
    peaks = [k for k in acf.index if (k == 1 or acf[k] > acf.get(k - 1, -1)) and acf[k] > acf.get(k + 1, -1)]
    out(f"- Local peaks (higher than both neighbours) at lags {', '.join(map(str, peaks))}; "
        f"lowest at lag(s) {', '.join(str(k) for k in acf.index if acf[k] == acf.min())}. "
        "Peaks at 7 and 14 reflect the weekly cycle; this supports lags 1, 2, 7 and 7/14-day rolling means.")

    # 8. Weather vs sales
    out("\n## 8. Weather vs sales (clean days)")
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    for a, c in zip(ax, ["avg_temperature", "precpt", "avg_humidity", "avg_wind_level"]):
        q = pd.qcut(clean[c], 8, duplicates="drop")
        m = clean.groupby(q, observed=True).rel.mean()
        a.plot(range(len(m)), m.values, marker="o", color="#54A24B")
        a.set(title=c, xlabel="octile (low -> high)", ylabel="relative sales")
        out(f"- {c}: relative sales from {m.iloc[0]:.2f} (lowest octile) to {m.iloc[-1]:.2f} (highest); "
            f"range {m.max() - m.min():.2f}")
    save(fig, "08_weather.png")

    # 9. Stockout rate by category and sales group
    out("\n## 9. Stockout rate by category and sales group")
    cat = tr.groupby("first_category_id").agg(rate=("is_stockout_day", "mean"), n=("series_id", "nunique"))
    cat = cat[cat.n >= 20].sort_values("rate")
    grp = tr.groupby("sales_group", observed=True).is_stockout_day.mean()
    out(f"- Stockout-day rate: low-sales {grp.get('low'):.1%}, high-sales {grp.get('high'):.1%}")
    out(f"- Across categories with >= 20 series: {cat.rate.min():.1%} to {cat.rate.max():.1%}")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    ax[0].bar(cat.index.astype(str), cat.rate, color="#E45756")
    ax[0].set(title="Stockout-day rate by first category (>= 20 series)", xlabel="first_category_id", ylabel="share of days")
    ax[1].bar(grp.index.astype(str), grp.values, color="#E45756")
    ax[1].set(title="By sales group")
    save(fig, "09_stockout_by_category.png")

    # 10. Train vs test shift
    out("\n## 10. Train vs test week (descriptive only, not used for any modelling choice)")
    daily = p.groupby("dt").agg(sales=("sale_amount", "mean"), temp=("avg_temperature", "mean"),
                                promo=("activity_flag", "mean"), stockout=("is_stockout_day", "mean"))
    last = daily.loc["2024-06-19":"2024-06-25"].mean()
    test = daily.loc["2024-06-26":].mean()
    for c in daily.columns:
        out(f"- {c}: last train week {last[c]:.3f} vs test week {test[c]:.3f}")
    fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    for a, c, col in zip(ax, daily.columns, ["#4C78A8", "#54A24B", "#F58518", "#E45756"]):
        a.plot(daily.index, daily[c], color=col)
        a.axvline(pd.Timestamp("2024-06-26"), color="k", ls="--", lw=.8)
        a.set_ylabel(c)
    ax[0].set_title("Daily averages across the sample (dashed line = test week starts)")
    save(fig, "10_train_vs_test.png")

    (OUT / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
