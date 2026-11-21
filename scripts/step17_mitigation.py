"""Step 17: alternative approaches to stockout censoring - which one should a practitioner use?

Starting from the censored data (condition B, 30% demand-linked masking), four ways of training are compared
against the clean reference A on every headline metric:

  1. naive          train on recorded sales as if they were demand (= B in Steps 6-11)
  2. drop           exclude stockout days as training targets
  3. impute         replace stockout-day sales with estimated full-day demand (hourly sales pattern)
  4. censored loss  treat stockout-day sales as a lower bound (one-sided squared loss)

Approaches 2-4 use what a real retailer knows (which days stocked out and when), never the true demand.
"Gap closed" = share of naive's distance to A that an approach removes (100% = as good as clean data).

Outputs: outputs/step17_mitigation_report.md, step17_results.csv, step17_gap_closed.png
Usage:   .\\.venv\\Scripts\\python.exe scripts\\step17_mitigation.py [--seeds 42 43] [--sample-size 5000]
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blindspot.conditions import ConditionConfig, build_conditions  # noqa: E402
from blindspot.data import PROJECT_ROOT  # noqa: E402
from blindspot.experiment import METRICS, RELATIVE_TO_A, worse  # noqa: E402
from blindspot.mitigation import APPROACHES, LABELS, imputation_quality, run_approaches  # noqa: E402
from blindspot.model import load_config  # noqa: E402
from blindspot.prepare import PROCESSED_DIR, nested_samples  # noqa: E402

OUT = PROJECT_ROOT / "outputs"
MAIN_SIZE = 5_000
KEY = ["wape", "bias", "row_spearman_vs_A", "discount_share_change", "activity_share_change", "injected_recovery",
       "cost_per_demand", "fill_rate", "promo_extra_cost"]
CORE = ["wape", "row_spearman_vs_A", "injected_recovery", "cost_per_demand", "promo_extra_cost"]
FIXES = [a for a in APPROACHES if a != "naive"]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--sample-size", type=int, default=MAIN_SIZE, help="smaller values are for quick trials")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(OUT / "models" / "lgb_config.json")
    tab = pd.read_parquet(PROCESSED_DIR / "series.parquet")
    base = pd.read_parquet(PROCESSED_DIR / "sample_series.parquet")
    ids = nested_samples(tab, base, [a.sample_size])[a.sample_size].series_id
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet", filters=[("series_id", "in", ids.tolist())])
    panel = panel.sort_values(["series_id", "dt"]).reset_index(drop=True)
    main_run = a.sample_size == MAIN_SIZE

    out("# Step 17 report: alternative approaches to stockout censoring\n")
    out(f"{a.sample_size:,} series, 30% demand-linked masking, seeds {', '.join(map(str, a.seeds))}. "
        "Every approach starts from the censored sales; approaches 2-4 also know which days stocked out and at "
        "what hour, but never the true demand.\n")
    out("| approach | what it does |")
    out("|---|---|")
    out("| 1. naive (= B) | trains on recorded sales as if they were demand |")
    out("| 2. drop stockout days | removes stockout days as training targets (they still feed the lag features) |")
    out("| 3. impute full-day demand | scales the sales before the stockout up by the series' usual share of the "
        "day's sales made by that hour; for very early stockouts, adds the usual rest of the day |")
    out("| 4. censored loss | keeps stockout days but treats their sales as a minimum: forecasts below are "
        "penalised, forecasts above are not |")
    out("\nApproaches 2-4 set the ordering safety allowance from validation days without a stockout.\n")

    # Imputation quality (possible only because this study knows the true demand)
    cond, _ = build_conditions(panel, ConditionConfig(seed=a.seeds[0]))
    q = imputation_quality(cond, panel)
    out(f"## How good is the imputation? (seed {a.seeds[0]}, {q['days']:,} simulated stockout days)\n")
    out("| sales on stockout days | total vs true demand | WAPE vs true demand |")
    out("|---|---|---|")
    out(f"| recorded (censored) | {q['censored_shortfall']:+.1%} | {q['censored_wape']:.3f} |")
    out(f"| imputed | {q['imputed_shortfall']:+.1%} | {q['imputed_wape']:.3f} |")
    out("")
    check(abs(q["imputed_shortfall"]) < abs(q["censored_shortfall"]), "imputation reduces the demand shortfall")

    # Runs
    parts, cache = [], {}
    for s in a.seeds:
        ts = time.time()
        parts.append(run_approaches(panel, s, cfg, cache))
        print(f"  seed {s}: {time.time() - ts:.0f}s", flush=True)
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(OUT / "step17_results.csv", index=False)

    if main_run:
        prev = pd.read_csv(OUT / "step11_seed_results.csv").replace({"condition": {"B": "naive"}})
        m = res.merge(prev, on=["seed", "condition", "metric"], suffixes=("", "_s11"))
        m = m[m.condition.isin(["A", "naive"])]
        check(len(m) > 0 and (m.value - m.value_s11).abs().max() < 1e-3,
              "A and naive reproduce Step 11's A and B for the shared seeds")

    wide = res.pivot_table(index=["seed", "metric"], columns="condition", values="value").reset_index()
    mean = wide.groupby("metric")[["A"] + APPROACHES].mean()
    n = len(a.seeds)

    out(f"\n## Results (mean over {n} seed{'s' if n > 1 else ''})\n")
    out("| metric | " + " | ".join(LABELS[c] for c in ["A"] + APPROACHES) + " |")
    out("|---" * (2 + len(APPROACHES)) + "|")
    for k in KEY:
        f = METRICS[k][1]
        out(f"| {METRICS[k][0]} | " + " | ".join("-" if c == "A" and k in RELATIVE_TO_A else f.format(mean.loc[k, c])
                                                  for c in ["A"] + APPROACHES) + " |")

    # Gap closed: share of naive's distance to A removed by each approach
    gap = {}
    for k in CORE:
        d0 = abs(mean.loc[k, "naive"] - mean.loc[k, "A"])
        gap[k] = {c: (1 - abs(mean.loc[k, c] - mean.loc[k, "A"]) / d0) if d0 > 0 else np.nan for c in FIXES}
    gap = pd.DataFrame(gap).T
    out("\n## Gap to clean data closed (100% = as good as A, 0% = no better than naive, negative = worse)\n")
    out("| metric | " + " | ".join(LABELS[c] for c in FIXES) + " |")
    out("|---" * (1 + len(FIXES)) + "|")
    for k in CORE:
        out(f"| {METRICS[k][0]} | " + " | ".join(f"{gap.loc[k, c]:.0%}" for c in FIXES) + " |")
    avg = gap.mean()
    out("| **average** | " + " | ".join(f"**{avg[c]:.0%}**" for c in FIXES) + " |")

    # Which approaches beat naive in every seed?
    out("\n## Better than naive in every seed?\n")
    out("| metric | " + " | ".join(LABELS[c] for c in FIXES) + " |")
    out("|---" * (1 + len(FIXES)) + "|")
    beats = {}
    for c in FIXES:
        beats[c] = {k: int(sum(worse(k, x.naive, getattr(x, c)) for x in wide[wide.metric == k].itertuples()))
                    for k in CORE}
    for k in CORE:
        out(f"| {METRICS[k][0]} | " + " | ".join(f"{beats[c][k]}/{n}" for c in FIXES) + " |")
    out("")
    best = avg.idxmax()
    for c in FIXES:
        verdict(all(beats[c][k] == n for k in CORE), f"{LABELS[c]} is better than naive on every core metric "
                "in every seed")
    out(f"\n## Recommendation\n\n**{LABELS[best]}** closes the largest share of the gap to clean data on average "
        f"({avg[best]:.0%}). Its promotion-effect recovery is {mean.loc['injected_recovery', best]:.0%} "
        f"(naive {mean.loc['injected_recovery', 'naive']:.0%}, clean {mean.loc['injected_recovery', 'A']:.0%}).")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.8 / len(FIXES)
    x = np.arange(len(CORE))
    for i, c in enumerate(FIXES):
        ax.bar(x + i * w - 0.4 + w / 2, gap[c].to_numpy() * 100, w, label=LABELS[c])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(100, color="grey", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([METRICS[k][0] for k in CORE], fontsize=8)
    ax.set_ylabel("gap to clean data closed (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "step17_gap_closed.png", dpi=110)
    plt.close(fig)
    out("\n![gap closed](step17_gap_closed.png)")

    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    (OUT / "step17_mitigation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {OUT / 'step17_mitigation_report.md'}")


if __name__ == "__main__":
    main()
