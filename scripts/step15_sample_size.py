"""Step 15: sample-size check - do the conclusions depend on studying 5,000 series?

Reruns the main experiment (30% demand-linked masking, conditions A, B, C, all headline metrics) on nested
stratified samples of 2,500, 5,000 and 10,000 series. The 2,500 sample is a subset of the study sample and the
10,000 sample contains it, so the runs differ in size only. The model configuration stays frozen (Step 6a).

The 5,000 run is read from Step 11 (same sample, same code) unless --rerun-main is given, in which case it is
recomputed and checked against Step 11.
Outcomes are reported as HOLDS / DOES NOT HOLD; they are findings, not code checks.

Outputs: outputs/step15_sample_size_report.md, step15_results.csv, step15_sample_size.png
Usage:   .\\.venv\\Scripts\\python.exe scripts\\step15_sample_size.py [--sizes 2500 5000 10000] [--seeds 42]
         [--rerun-main]
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from blindspot.data import PROJECT_ROOT  # noqa: E402
from blindspot.experiment import METRICS, RELATIVE_TO_A, run_seed, worse  # noqa: E402
from blindspot.model import load_config  # noqa: E402
from blindspot.prepare import PROCESSED_DIR, nested_samples  # noqa: E402

OUT = PROJECT_ROOT / "outputs"
COLORS = {"A": "#4C78A8", "B": "#E45756", "C": "#F58518"}
MAIN_SIZE = 5_000
KEY = ["wape", "bias", "row_spearman_vs_A", "discount_share_change", "activity_share_change", "injected_recovery",
       "cost_per_demand", "promo_extra_cost"]
CORE = ["wape", "row_spearman_vs_A", "injected_recovery", "cost_per_demand", "promo_extra_cost"]
lines: list[str] = []
problems: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def check(ok: bool, msg: str) -> None:
    out(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        problems.append(msg)


def verdict(ok: bool, msg: str) -> None:
    out(f"- **{'HOLDS' if ok else 'DOES NOT HOLD'}**: {msg}")


def load_series_panel(ids: pd.Series) -> pd.DataFrame:
    """Read only the rows of the given series from the full panel."""
    p = pd.read_parquet(PROCESSED_DIR / "panel.parquet", filters=[("series_id", "in", ids.tolist())])
    return p.sort_values(["series_id", "dt"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[2_500, 5_000, 10_000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--rerun-main", action="store_true", help="recompute the 5,000 run instead of reading Step 11")
    a = ap.parse_args()
    sizes = sorted(a.sizes)
    t0 = time.time()
    cfg = load_config(OUT / "models" / "lgb_config.json")
    tab = pd.read_parquet(PROCESSED_DIR / "series.parquet")
    base = pd.read_parquet(PROCESSED_DIR / "sample_series.parquet")

    out("# Step 15 report: sample-size check\n")
    out(f"Does the main finding depend on the {MAIN_SIZE:,}-series sample? Same experiment (30% demand-linked "
        "masking), same frozen model configuration, nested stratified samples.\n")

    # Samples
    samples = nested_samples(tab, base, sizes)
    pool = tab[tab.eligible]
    out("## Samples\n")
    out("| sample | series | rows | share low-sales | categories | mean stockout rate (train) | mean promo rate (train) |")
    out("|---|---|---|---|---|---|---|")
    out(f"| eligible pool | {len(pool):,} | - | {(pool.sales_group == 'low').mean():.1%} | "
        f"{pool.first_category_id.nunique()} | {pool.stockout_rate_train.mean():.1%} | {pool.promo_rate_train.mean():.1%} |")
    for n, s in samples.items():
        out(f"| {n:,} | {len(s):,} | {len(s) * 97:,} | {(s.sales_group == 'low').mean():.1%} | "
            f"{s.first_category_id.nunique()} | {s.stockout_rate_train.mean():.1%} | {s.promo_rate_train.mean():.1%} |")
    out("")
    for n, s in samples.items():
        check(len(s) == n and s.series_id.is_unique and s.eligible.all(), f"{n:,}: exact size, unique, all eligible")
        check(abs((s.sales_group == "low").mean() - (pool.sales_group == "low").mean()) < 0.01,
              f"{n:,}: low-sales share within 1 point of the eligible pool")
    for small, big in zip(sizes, sizes[1:]):
        check(samples[small].series_id.isin(samples[big].series_id).all(), f"{small:,} sample is inside the {big:,} sample")
    if MAIN_SIZE in samples:
        check(samples[MAIN_SIZE].series_id.tolist() == sorted(base.series_id), f"{MAIN_SIZE:,} sample is the study sample")

    # Runs
    prev = pd.read_csv(OUT / "step11_seed_results.csv")
    parts = []
    for n in sizes:
        cache: dict = {}
        panel = None
        for s in a.seeds:
            if n == MAIN_SIZE and s in prev.seed.values and not a.rerun_main:
                parts.append(prev[prev.seed == s].assign(size=n))
                continue
            if panel is None:
                panel = load_series_panel(samples[n].series_id)
                check(panel.series_id.nunique() == n and panel.groupby("series_id").size().eq(97).all(),
                      f"{n:,}: panel has every series with all 97 days")
            ts = time.time()
            r = run_seed(panel, s, cfg, cache).assign(size=n)
            parts.append(r)
            print(f"  {n:,} series, seed {s}: {time.time() - ts:.0f}s", flush=True)
            if n == MAIN_SIZE and s in prev.seed.values:
                m = r.merge(prev[prev.seed == s], on=["seed", "condition", "metric"], suffixes=("", "_s11"))
                check((m.value - m.value_s11).abs().max() < 1e-3, f"rerun of {n:,}, seed {s} matches Step 11")
        del panel
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(OUT / "step15_results.csv", index=False)

    wide = res.pivot_table(index=["size", "seed", "metric"], columns="condition", values="value").reset_index()
    mean = wide.groupby(["size", "metric"])[["A", "B", "C"]].mean()
    n_seeds = len(a.seeds)

    out(f"\n## Results (mean over seed{'s' if n_seeds > 1 else ''} {', '.join(map(str, a.seeds))})\n")
    out("| metric | " + " | ".join(f"{c}: {n:,}" for c in "ABC" for n in sizes) + " |")
    out("|---" * (1 + 3 * len(sizes)) + "|")
    for k in KEY:
        f = METRICS[k][1]
        cells = []
        for c in "ABC":
            for n in sizes:
                cells.append("-" if c == "A" and k in RELATIVE_TO_A else f.format(mean.loc[(n, k), c]))
        out(f"| {METRICS[k][0]} | " + " | ".join(cells) + " |")

    # Verdicts
    out("\n## Does the finding hold at every size?\n")
    out("| metric | " + " | ".join(f"B worse than C: {n:,}" for n in sizes) + " | "
        + " | ".join(f"B worse than A: {n:,}" for n in sizes) + " |")
    out("|---" * (1 + 2 * len(sizes)) + "|")
    bc, ba = {}, {}
    for k in KEY:
        bc[k] = {n: int(sum(worse(k, x.B, x.C) for x in wide[(wide["size"] == n) & (wide.metric == k)].itertuples()))
                 for n in sizes}
        ba[k] = {n: (None if k in RELATIVE_TO_A else
                     int(sum(worse(k, x.B, x.A) for x in wide[(wide["size"] == n) & (wide.metric == k)].itertuples())))
                 for n in sizes}
        out(f"| {METRICS[k][0]} | " + " | ".join(f"{bc[k][n]}/{n_seeds}" for n in sizes) + " | "
            + " | ".join("-" if ba[k][n] is None else f"{ba[k][n]}/{n_seeds}" for n in sizes) + " |")
    out("")
    verdict(all(bc[k][n] == n_seeds for k in CORE for n in sizes),
            "at every sample size, censoring is worse than matched noise on forecast error, explanation agreement, "
            "promotion-effect recovery and ordering costs")
    verdict(all(ba[k][n] == n_seeds for k in CORE if k not in RELATIVE_TO_A for n in sizes),
            "at every sample size, censoring is worse than the clean model on the same metrics")
    rec = {n: mean.loc[(n, "injected_recovery")] for n in sizes}
    spread = max(r.B for r in rec.values()) - min(r.B for r in rec.values())
    verdict(spread < 0.10, f"B's promotion-effect recovery varies by less than 10 points across sizes "
            f"({', '.join(f'{n:,}: {rec[n].B:.0%}' for n in sizes)}; A: "
            f"{', '.join(f'{rec[n].A:.0%}' for n in sizes)})")

    # Plot
    ks = ["wape", "row_spearman_vs_A", "injected_recovery", "cost_per_demand"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, k in zip(axs, ks):
        for c in "ABC":
            if c == "A" and k in RELATIVE_TO_A:
                continue
            ax.plot(sizes, [mean.loc[(n, k), c] for n in sizes], marker="o", color=COLORS[c], label=c)
        ax.set_title(METRICS[k][0], fontsize=9)
        ax.set_xlabel("series in sample")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{n:,}" for n in sizes])
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "step15_sample_size.png", dpi=110)
    plt.close(fig)
    out("\n![sample size](step15_sample_size.png)")

    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    (OUT / "step15_sample_size_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {OUT / 'step15_sample_size_report.md'}")


if __name__ == "__main__":
    main()
