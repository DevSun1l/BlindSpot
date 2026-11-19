"""Build clean (A), stockout-censored (B), and matched-noise (C) training conditions.

Outputs:
  data/processed/conditions.parquet   per-series-day sales under A/B/C and ground truth
  outputs/step3_conditions_report.md
  outputs/step3_conditions.png
Usage:
    .\\.venv\\Scripts\\python.exe scripts\\step3_build_conditions.py [--mask-rate 0.3] [--demand-power 1.0]
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from blindspot.conditions import ConditionConfig, build_conditions  # noqa: E402
from blindspot.data import PROJECT_ROOT  # noqa: E402
from blindspot.prepare import PROCESSED_DIR, load_panel  # noqa: E402

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
    ap.add_argument("--mask-rate", type=float, default=ConditionConfig.mask_rate)
    ap.add_argument("--demand-power", type=float, default=ConditionConfig.demand_power)
    ap.add_argument("--seed", type=int, default=ConditionConfig.seed)
    a = ap.parse_args()
    cfg = ConditionConfig(mask_rate=a.mask_rate, demand_power=a.demand_power, seed=a.seed)

    panel = load_panel(sample_only=True)
    df, start_dist = build_conditions(panel, cfg)

    out("# Step 3 report: training-data conditions\n")
    out(f"Config: `{json.dumps(cfg.as_dict())}`\n")

    m = df.is_masked == 1
    tr = df.split == "train"
    cand = tr & (df.is_clean_day == 1)
    out("## Masking\n")
    out(f"- Candidate days (clean train days): {cand.sum():,}; masked: {m.sum():,} "
        f"({m.sum() / cand.sum():.1%} of candidates, target {cfg.mask_rate:.0%})")
    changed = m & ~np.isclose(df.sales_B, df.sales_true)
    out(f"- Effective censoring: {changed.sum():,} masked days ({changed.sum() / cand.sum():.1%} of candidates) actually "
        f"lose sales; the other {(m & ~changed).sum() / m.sum():.1%} of masked days had no sales after the simulated "
        "stockout hour, so censoring leaves them unchanged (in B and C alike).")
    per_series = df[cand].groupby("series_id").is_masked.mean()
    out(f"- Masked share per series: median {per_series.median():.1%}, 5th-95th pct "
        f"{per_series.quantile(.05):.1%} to {per_series.quantile(.95):.1%}")
    check(abs(m.sum() / cand.sum() - cfg.mask_rate) < 0.01, "overall masked share within 1 point of target")
    check(not (m & ~cand).any(), "only clean train days are masked (no stockout days, no test days)")

    rel = df.sales_true / df[cand].groupby("series_id").sales_true.transform("mean")
    out(f"- Demand link: masked days had true sales {rel[m].mean():.2f}x their series' clean-day mean; "
        f"unmasked clean days {rel[cand & ~m].mean():.2f}x")
    if cfg.demand_power > 0:
        check(rel[m].mean() > rel[cand & ~m].mean(), "masked days are higher-demand days on average")

    out("\n## Simulated stockout hour vs real pattern\n")
    sim = df.loc[m, "sim_stockout_hour"].value_counts(normalize=True).sort_index()
    tvd = 0.5 * (sim.reindex(start_dist.index, fill_value=0) - start_dist).abs().sum()
    out(f"- Total variation distance between simulated and real start-hour distributions: {tvd:.3f}")
    check(tvd < 0.02, "simulated stockout hours follow the real distribution (TVD < 0.02)")

    out("\n## Distortion on masked days (truth = condition A)\n")
    t, b, c = df.loc[m, "sales_true"], df.loc[m, "sales_B"], df.loc[m, "sales_C"]
    out("")
    out("| condition | mean abs. distortion | relative (sum abs / sum true) | bias (sum signed / sum true) | days changed |")
    out("|---|---|---|---|---|")
    for name, v in [("A clean", df.loc[m, "sales_A"]), ("B censored", b), ("C matched noise", c)]:
        out(f"| {name} | {np.abs(v - t).mean():.3f} | {np.abs(v - t).sum() / t.sum():.1%} | "
            f"{(v - t).sum() / t.sum():+.1%} | {(v != t).mean():.1%} |")
    check((df.sales_A == df.sales_true).all(), "A equals the truth everywhere")
    check((df.sales_B <= df.sales_true + 1e-9).all(), "B never exceeds the truth (censoring is one-sided)")
    check(np.allclose(np.abs(c - t), np.abs(b - t)), "C has exactly the same absolute distortion as B on every masked day")
    check((df.sales_C >= -1e-9).all(), "C is never negative")
    check(abs((c - t).sum() / t.sum()) < 0.03, "C is roughly unbiased (|bias| < 3%)")
    unchanged = ~m
    check((df.loc[unchanged, ["sales_A", "sales_B", "sales_C"]].sub(df.loc[unchanged, "sales_true"], axis=0).abs() < 1e-12).all().all(),
          "all unmasked days (real stockout days and the test week) are identical in A, B and C")

    out("\n## Effect on the whole training set\n")
    for col in ["sales_A", "sales_B", "sales_C"]:
        out(f"- {col}: total train sales {df.loc[tr, col].sum() / df.loc[tr, 'sales_A'].sum():.1%} of A; "
            f"zero-sale days {(df.loc[tr, col] == 0).mean():.1%}")

    # Diagnostic summary plots: start-hour match, relative distortion, and an example series.
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].bar(start_dist.index - 0.2, start_dist.values, width=0.4, label="real", color="#E45756")
    ax[0].bar(sim.index + 0.2, sim.values, width=0.4, label="simulated", color="#4C78A8")
    ax[0].set(title="Stockout start hour", xlabel="hour", xticks=range(6, 22))
    ax[0].legend()
    ax[1].hist([(b - t) / t.clip(lower=1e-9), (c - t) / t.clip(lower=1e-9)], bins=40, label=["B censored", "C noise"],
               color=["#E45756", "#F58518"])
    ax[1].set(title="Relative distortion on masked days", xlabel="(condition - true) / true")
    ax[1].legend()
    sid = df.loc[m].series_id.value_counts().index[0]
    s = df[(df.series_id == sid) & tr]
    ax[2].plot(s.dt, s.sales_A, label="A clean", color="#4C78A8", lw=1.5)
    ax[2].plot(s.dt, s.sales_B, label="B censored", color="#E45756", lw=1)
    ax[2].plot(s.dt, s.sales_C, label="C noise", color="#F58518", lw=1, ls="--")
    ax[2].scatter(s.dt[s.is_masked == 1], s.sales_A[s.is_masked == 1], s=12, color="k", label="masked day", zorder=3)
    ax[2].set(title=f"Example series {sid} (train days)")
    ax[2].legend(fontsize=8)
    ax[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "outputs" / "step3_conditions.png", dpi=110)
    out("\n![chart](step3_conditions.png)")

    df.to_parquet(PROCESSED_DIR / "conditions.parquet", index=False)
    (PROCESSED_DIR / "conditions_config.json").write_text(json.dumps(cfg.as_dict(), indent=2))
    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    (PROJECT_ROOT / "outputs" / "step3_conditions_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
