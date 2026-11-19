"""Step 11: experiment runner.

1. Pipeline: re-runs Steps 3-10 in order (Step 6a tuning only with --retune), each in its own process, logging
   to outputs/logs/. Stops at the first step that crashes or reports a failed check.
2. Robustness: re-runs the whole A/B/C experiment in memory for several masking seeds (default 42 43 44 45)
   and collects the headline numbers into one table. Seed 42 is the main run; its in-memory numbers are checked
   against the files Steps 6-10 wrote, so the runner provably measures the same thing.

Outputs: outputs/step11_experiment_report.md, step11_results_table.csv, step11_seed_results.csv, step11_seeds.png,
         outputs/logs/<step>.log
Usage:   .\\.venv\\Scripts\\python.exe scripts\\step11_experiment.py [--skip-pipeline] [--retune] [--seeds 42 43 44 45]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blindspot.data import PROJECT_ROOT  # noqa: E402
from blindspot.experiment import METRICS, RELATIVE_TO_A, consistency, run_seed  # noqa: E402
from blindspot.explain import local_stability  # noqa: E402
from blindspot.features import FEATURES  # noqa: E402
from blindspot.model import load_config  # noqa: E402
from blindspot.prepare import PROCESSED_DIR, load_panel  # noqa: E402

OUT = PROJECT_ROOT / "outputs"
LOGS = OUT / "logs"
SCRIPTS = PROJECT_ROOT / "scripts"
PIPELINE = ["step3_build_conditions", "step4_build_features", "step5_baselines", "step6b_train", "step7_shap",
            "step8_faithfulness", "step9_whatif", "step10_inventory"]
COLORS = {"A": "#4C78A8", "B": "#E45756", "C": "#F58518"}
lines: list[str] = []
problems: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def check(ok: bool, msg: str) -> None:
    out(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        problems.append(msg)


def run_pipeline(retune: bool) -> bool:
    out("## 1. Pipeline (Steps 3-10)\n")
    LOGS.mkdir(exist_ok=True)
    steps = PIPELINE[:3] + (["step6a_tune"] if retune else []) + PIPELINE[3:]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    out("| step | status | time | log |")
    out("|---|---|---|---|")
    for s in steps:
        t0 = time.time()
        print(f"  running {s} ...", flush=True)
        r = subprocess.run([sys.executable, str(SCRIPTS / f"{s}.py")], cwd=PROJECT_ROOT, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        log = LOGS / f"{s}.log"
        log.write_text(r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr.strip() else ""), encoding="utf-8")
        ok = r.returncode == 0 and "FAILED CHECKS" not in r.stdout and "[FAIL]" not in r.stdout
        out(f"| {s} | {'OK' if ok else 'FAILED'} | {time.time() - t0:.0f}s | logs/{s}.log |")
        if not ok:
            out(f"\nStopped: {s} failed (exit code {r.returncode}). See outputs/logs/{s}.log.")
            if "PermissionError" in r.stderr:
                out("A file it writes is locked, usually because it is open in Excel or another program. "
                    "Close it and run again: " + r.stderr.strip().splitlines()[-1])
            problems.append(f"pipeline step {s} failed")
            return False
    out("")
    check(True, f"all {len(steps)} pipeline steps ran and reported no failed checks")
    return True


def check_against_steps(res: pd.DataFrame) -> None:
    """The in-memory seed-42 run must reproduce the numbers Steps 6, 7 and 10 saved."""
    v = res[res.seed == 42].set_index(["condition", "metric"]).value
    s6 = pd.read_csv(OUT / "step6_results.csv")
    s6 = s6[s6.group == "all"]
    for c in "ABC":
        ref = s6[s6.model.str.startswith(f"LightGBM {c}")].wape.iloc[0]
        check(abs(v[(c, "wape")] - ref) < 1e-3, f"seed 42 WAPE {c} = Step 6 ({ref:.4f})")
    keys = ["series_id", "dt"]
    sv = {c: pd.read_parquet(PROCESSED_DIR / f"shap_{c}.parquet") for c in "ABC"}
    for c in "BC":
        ref = local_stability(sv["A"][FEATURES], sv[c][FEATURES]).spearman.mean()
        # tolerance 1e-3: the saved SHAP files are stored at lower precision, which can swap near-tied ranks
        check(abs(v[(c, "row_spearman_vs_A")] - ref) < 1e-3, f"seed 42 per-row SHAP agreement A vs {c} = Step 7 ({ref:.3f})")
    assert all((sv["A"][k] == sv[c][k]).all() for c in "BC" for k in keys)
    s10 = pd.read_csv(OUT / "step10_results.csv")
    e, p = s10[s10.part == "everyday"].set_index("model"), s10[s10.part == "promo"].set_index("model")
    for c in "ABC":
        ok = (abs(v[(c, "cost_per_demand")] - e.loc[c, "cost_per_demand"]) < 1e-9
              and abs(v[(c, "injected_recovery")] - p.loc[c, "rec"]) < 1e-9
              and abs(v[(c, "promo_extra_cost")] - p.loc[c, "extra"]) < 1e-9)
        check(ok, f"seed 42 ordering cost, injected recovery and promo extra cost for {c} = Steps 9-10")


def fmt(k: str, x: float) -> str:
    return "-" if pd.isna(x) else METRICS[k][1].format(x)


def report_tables(res: pd.DataFrame, table: pd.DataFrame, seeds: list[int]) -> None:
    main = res[res.seed == seeds[0]].pivot_table(index="metric", columns="condition", values="value")
    out(f"\n## 3. Main results (masking seed {seeds[0]})\n")
    out("| metric | A (clean) | B (censored) | C (matched noise) |")
    out("|---|---|---|---|")
    for k, (label, _, _) in METRICS.items():
        a = "reference" if k in RELATIVE_TO_A else fmt(k, main.loc[k, "A"])
        out(f"| {label} | {a} | {fmt(k, main.loc[k, 'B'])} | {fmt(k, main.loc[k, 'C'])} |")
    out("\nNoise floor for SHAP agreement (Step 7): retraining A with another model seed gives per-row agreement 0.944.")

    n = len(seeds)
    out(f"\n## 4. Robustness across {n} masking seeds ({', '.join(map(str, seeds))})\n")
    out("Each seed redraws which training days are masked, the simulated stockout hours, and C's noise signs. "
        "Condition A does not depend on the masking seed. 'Worse' follows each metric's direction "
        "(higher WAPE/cost, larger |bias| or |share change|, lower agreement/fill rate, further from 100% recovery).\n")
    out("| metric | A | B mean (min to max) | C mean (min to max) | B worse than A | B worse than C |")
    out("|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        k = r.metric
        a = "reference" if k in RELATIVE_TO_A else fmt(k, r.A_mean)
        ba = "n/a" if pd.isna(r.B_worse_than_A) else f"{int(r.B_worse_than_A)}/{n}"
        out(f"| {METRICS[k][0]} | {a} | {fmt(k, r.B_mean)} ({fmt(k, r.B_min)} to {fmt(k, r.B_max)}) | "
            f"{fmt(k, r.C_mean)} ({fmt(k, r.C_min)} to {fmt(k, r.C_max)}) | {ba} | {int(r.B_worse_than_C)}/{n} |")
    out("")
    hold_a = table.dropna(subset=["B_worse_than_A"])
    all_a = hold_a[hold_a.B_worse_than_A == n].metric.tolist()
    all_c = table[table.B_worse_than_C == n].metric.tolist()
    out(f"- B is worse than A in every seed on {len(all_a)} of {len(hold_a)} absolute metrics"
        + (f"; not in every seed: {', '.join(sorted(set(hold_a.metric) - set(all_a)))}." if len(all_a) < len(hold_a) else "."))
    out(f"- B is worse than C (matched noise) in every seed on {len(all_c)} of {len(table)} metrics"
        + (f"; not in every seed: {', '.join(sorted(set(table.metric) - set(all_c)))}." if len(all_c) < len(table) else "."))


def plot(res: pd.DataFrame) -> None:
    keys = ["wape", "row_spearman_vs_A", "discount_share_change", "injected_recovery", "cost_per_demand",
            "promo_extra_cost"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    for a, k in zip(ax.ravel(), keys):
        for i, c in enumerate("ABC"):
            v = res[(res.metric == k) & (res.condition == c)].value.to_numpy()
            a.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=COLORS[c], s=30)
            a.hlines(v.mean(), i - 0.25, i + 0.25, color=COLORS[c], lw=2)
        a.set_xticks(range(3), list("ABC"))
        a.set_title(METRICS[k][0], fontsize=9)
    fig.suptitle("Headline metrics per masking seed (dots) and their mean (bar)")
    fig.tight_layout()
    fig.savefig(OUT / "step11_seeds.png", dpi=110)
    plt.close(fig)
    out("\n![seeds](step11_seeds.png)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pipeline", action="store_true", help="only run the multi-seed robustness part")
    ap.add_argument("--retune", action="store_true", help="also re-run Step 6a hyperparameter tuning (slow)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--explain-rows", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    out("# Step 11 report: experiment runner\n")
    if not a.skip_pipeline and not run_pipeline(a.retune):
        finish(t0)
        return
    if a.skip_pipeline:
        out("## 1. Pipeline\n\nSkipped (--skip-pipeline); using the existing Step 3-10 outputs.\n")

    out(f"## 2. Multi-seed experiment\n")
    cfg = load_config(OUT / "models" / "lgb_config.json")
    panel = load_panel(sample_only=True)
    cache: dict = {}
    parts = []
    for s in a.seeds:
        ts = time.time()
        parts.append(run_seed(panel, s, cfg, cache, explain_rows=a.explain_rows))
        print(f"  seed {s} done in {time.time() - ts:.0f}s", flush=True)
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(OUT / "step11_seed_results.csv", index=False)
    out(f"Ran {len(a.seeds)} seeds with the frozen Step 6 configuration, {a.explain_rows:,} explained rows, "
        "promo scenario 'activity + 20% discount', injected +30% promo on 20% of days, newsvendor costs 1 / 0.5.\n")
    if 42 in a.seeds:
        check_against_steps(res)
    table = consistency(res)
    table.to_csv(OUT / "step11_results_table.csv", index=False)
    report_tables(res, table, a.seeds)
    plot(res)
    finish(t0)


def finish(t0: float) -> None:
    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    report = OUT / "step11_experiment_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
