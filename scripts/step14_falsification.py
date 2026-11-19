"""Step 14: falsification checks - try to break the main finding.

Main finding: one-sided censoring (B) distorts forecasts, explanations and decisions more than matched noise (C).
Two ways it could be an artefact of the setup:

1. Demand-linked masking. In the main run, high-demand days are masked more often (as with real stockouts).
   Check: mask days at RANDOM (demand_power = 0). If B no longer differs from C, the effect came from which
   days were masked, not from censoring itself.
2. No dose-response. If censoring is the cause, more censoring should do more damage.
   Check: mask 10%, 30% (main) and 50% of clean training days.

Each new setting runs on masking seeds 42 and 43; the main setting (30%, demand-linked) is read from Step 11.
Outcomes are reported as HOLDS / DOES NOT HOLD; they are findings, not code checks.

Outputs: outputs/step14_falsification_report.md, step14_results.csv, step14_dose_response.png
Usage:   .\\.venv\\Scripts\\python.exe scripts\\step14_falsification.py [--seeds 42 43]
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
from blindspot.experiment import METRICS, run_seed, worse  # noqa: E402
from blindspot.model import load_config  # noqa: E402
from blindspot.prepare import load_panel  # noqa: E402

OUT = PROJECT_ROOT / "outputs"
COLORS = {"A": "#4C78A8", "B": "#E45756", "C": "#F58518"}
SETTINGS = {  # name -> (mask_rate, demand_power)
    "random 30%": (0.30, 0.0),
    "demand-linked 10%": (0.10, 1.0),
    "demand-linked 30% (main)": (0.30, 1.0),
    "demand-linked 50%": (0.50, 1.0),
}
MAIN = "demand-linked 30% (main)"
KEY = ["wape", "bias", "row_spearman_vs_A", "discount_share_change", "activity_share_change", "injected_recovery",
       "cost_per_demand", "promo_extra_cost"]
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
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(OUT / "models" / "lgb_config.json")
    panel = load_panel(sample_only=True)

    out("# Step 14 report: falsification checks\n")
    out("Can the main finding (censoring hurts more than matched noise) be explained away by the setup?\n")

    # Realised mask shares (the 50% setting can fall short: probabilities are capped at 1).
    out("## Settings\n")
    out("| setting | target mask rate | masking | realised share of clean training days masked (seed 42) |")
    out("|---|---|---|---|")
    shares = {}
    for name, (rate, power) in SETTINGS.items():
        cond, _ = build_conditions(panel, ConditionConfig(mask_rate=rate, demand_power=power, seed=42))
        cand = (cond.split == "train") & (cond.is_clean_day == 1)
        shares[name] = cond.loc[cand, "is_masked"].mean()
        out(f"| {name} | {rate:.0%} | {'random' if power == 0 else 'demand-linked'} | {shares[name]:.1%} |")
    out("")
    for name, (rate, _) in SETTINGS.items():
        check(abs(shares[name] - rate) < 0.03, f"{name}: realised mask share within 3 points of target")

    # Runs
    prev = pd.read_csv(OUT / "step11_seed_results.csv")
    parts, cache = [], {}
    for name, (rate, power) in SETTINGS.items():
        for s in a.seeds:
            if name == MAIN and s in prev.seed.values:
                parts.append(prev[prev.seed == s].assign(setting=name))
                continue
            ts = time.time()
            parts.append(run_seed(panel, s, cfg, cache, mask_rate=rate, demand_power=power).assign(setting=name))
            print(f"  {name}, seed {s}: {time.time() - ts:.0f}s", flush=True)
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(OUT / "step14_results.csv", index=False)
    a_vals = res[res.condition == "A"].groupby(["metric"]).value.agg(["min", "max"])
    check(((a_vals["max"] - a_vals["min"]).abs() < 1e-9).all(), "condition A is identical in every setting and seed")

    wide = res.pivot_table(index=["setting", "seed", "metric"], columns="condition", values="value").reset_index()
    mean = wide.groupby(["setting", "metric"])[["A", "B", "C"]].mean()

    out(f"\n## Results (mean over seeds {', '.join(map(str, a.seeds))})\n")
    out("| metric | A | " + " | ".join(f"B: {n}" for n in SETTINGS) + " | " + " | ".join(f"C: {n}" for n in SETTINGS) + " |")
    out("|---" * (2 + 2 * len(SETTINGS)) + "|")
    for k in KEY:
        f = METRICS[k][1]
        a_txt = "-" if k in ("row_spearman_vs_A", "discount_share_change", "activity_share_change") else f.format(mean.loc[(MAIN, k), "A"])
        out(f"| {METRICS[k][0]} | {a_txt} | " + " | ".join(f.format(mean.loc[(n, k), "B"]) for n in SETTINGS) + " | "
            + " | ".join(f.format(mean.loc[(n, k), "C"]) for n in SETTINGS) + " |")

    # 1. Random masking
    out("\n## 1. Random masking: is censoring itself the cause?\n")
    r = wide[wide.setting == "random 30%"]
    b_vs_c = {k: int(sum(worse(k, x.B, x.C) for x in r[r.metric == k].itertuples())) for k in KEY}
    n = len(a.seeds)
    out("| metric | B worse than C (seeds) |")
    out("|---|---|")
    for k in KEY:
        out(f"| {METRICS[k][0]} | {b_vs_c[k]}/{n} |")
    out("")
    core = ["wape", "row_spearman_vs_A", "injected_recovery", "cost_per_demand", "promo_extra_cost"]
    verdict(all(b_vs_c[k] == n for k in core),
            "with masking unrelated to demand, censoring is still worse than matched noise on forecast error, "
            "explanation agreement, promotion-effect recovery and ordering costs in every seed")
    rec_r, rec_m = mean.loc[("random 30%", "injected_recovery"), "B"], mean.loc[(MAIN, "injected_recovery"), "B"]
    out(f"- B's promotion-effect recovery: {rec_r:.0%} with random masking vs {rec_m:.0%} with demand-linked masking "
        f"(A: {mean.loc[(MAIN, 'injected_recovery'), 'A']:.0%}). Part of the damage comes from censoring as such, part "
        "from censoring falling on high-demand (often promotion) days.")

    # 2. Dose-response
    out("\n## 2. Dose-response: does more censoring do more damage?\n")
    dose = ["demand-linked 10%", MAIN, "demand-linked 50%"]
    out("| metric | B - A at 10% | at 30% | at 50% | C - A at 10% | at 30% | at 50% | B gap grows with rate |")
    out("|---|---|---|---|---|---|---|---|")
    mono = {}
    for k in ["wape", "bias", "injected_recovery", "cost_per_demand", "promo_extra_cost"]:
        gb = [mean.loc[(d, k), "B"] - mean.loc[(d, k), "A"] for d in dose]
        gc = [mean.loc[(d, k), "C"] - mean.loc[(d, k), "A"] for d in dose]
        mono[k] = abs(gb[0]) < abs(gb[1]) < abs(gb[2])
        f = "{:+.3f}" if k != "injected_recovery" else "{:+.0%}"
        out(f"| {METRICS[k][0]} | " + " | ".join(f.format(x) for x in gb) + " | " + " | ".join(f.format(x) for x in gc)
            + f" | {'yes' if mono[k] else 'no'} |")
    out("")
    verdict(all(mono.values()), "the damage from censoring grows steadily with the amount of censoring on every "
            "metric shown")
    gap_c = [abs(mean.loc[(d, "injected_recovery"), "C"] - mean.loc[(d, "injected_recovery"), "A"]) for d in dose]
    gap_b = [abs(mean.loc[(d, "injected_recovery"), "B"] - mean.loc[(d, "injected_recovery"), "A"]) for d in dose]
    verdict(all(b > c for b, c in zip(gap_b, gap_c)),
            "at every censoring rate, B's promotion-effect error is larger than C's")

    # plot
    rates = [0.10, 0.30, 0.50]
    ks = ["wape", "injected_recovery", "cost_per_demand", "promo_extra_cost"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, k in zip(axs, ks):
        for c in "ABC":
            ax.plot(rates, [mean.loc[(d, k), c] for d in dose], marker="o", color=COLORS[c], label=c)
            if c != "A":
                ax.scatter([0.30], [mean.loc[("random 30%", k), c]], marker="x", s=70, color=COLORS[c])
        ax.set_title(METRICS[k][0], fontsize=9)
        ax.set_xlabel("share of clean training days masked")
        ax.set_xticks(rates)
    axs[0].legend(title="lines: demand-linked\nx: random 30%", fontsize=8, title_fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "step14_dose_response.png", dpi=110)
    plt.close(fig)
    out("\n![dose response](step14_dose_response.png)")

    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    (OUT / "step14_falsification_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {OUT / 'step14_falsification_report.md'}")


if __name__ == "__main__":
    main()
