"""Step 6a: tune LightGBM on condition A's validation days; compare gbdt vs dart.

Paired design: each sampled hyperparameter set is trained once with gbdt and once with dart,
so the two boosting types are compared on equal terms. The best configuration overall is frozen
for A, B and C. Tie rule (agreed): if dart's best is not better than gbdt's best by more than
TIE_MARGIN, gbdt is chosen (faster, supports early stopping).

Outputs:
  outputs/models/lgb_config.json    frozen configuration (params + number of rounds)
  outputs/step6a_tuning.csv         every trial
  outputs/step6a_tuning_report.md
Usage:  .\\.venv\\Scripts\\python.exe scripts\\step6a_tune.py [--trials 20]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from blindspot.data import PROJECT_ROOT
from blindspot.model import SEED, fit_and_validate, load_features, save_config, split_rows

TIE_MARGIN = 0.002
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def sample_params(rng: np.random.Generator) -> dict:
    return {
        "objective": str(rng.choice(["regression", "tweedie"])),
        "learning_rate": float(rng.choice([0.03, 0.05, 0.1])),
        "num_leaves": int(rng.choice([15, 31, 63, 127])),
        "min_child_samples": int(rng.choice([20, 50, 100, 200])),
        "feature_fraction": float(rng.choice([0.6, 0.8, 1.0])),
        "bagging_fraction": float(rng.choice([0.7, 0.85, 1.0])),
        "lambda_l2": float(rng.choice([0.0, 1.0, 10.0])),
        "n_estimators": int(rng.choice([300, 600, 1000])),   # used by dart only
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    fit, valid, _, _ = split_rows(load_features("A"))
    rng = np.random.default_rng(SEED)

    rows = []
    for i in range(a.trials):
        p = sample_params(rng)
        for boosting in ["gbdt", "dart"]:
            ts = time.time()
            wape, n = fit_and_validate({**p, "boosting": boosting}, fit, valid)
            rows.append({"trial": i, "boosting": boosting, **p, "rounds": n, "valid_wape": wape,
                         "seconds": round(time.time() - ts, 1)})
            print(f"trial {i:2d} {boosting:4s} wape={wape:.4f} rounds={n:4d} ({rows[-1]['seconds']}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(PROJECT_ROOT / "outputs" / "step6a_tuning.csv", index=False)

    best = res.loc[res.groupby("boosting").valid_wape.idxmin()].set_index("boosting")
    g, d = best.loc["gbdt"], best.loc["dart"]
    chosen = d if d.valid_wape < g.valid_wape - TIE_MARGIN else g
    chosen_name = "dart" if chosen is d else "gbdt"
    paired = res.pivot(index="trial", columns="boosting", values="valid_wape")
    dart_wins = int((paired.dart < paired.gbdt).sum())

    params = {k: (chosen[k].item() if hasattr(chosen[k], "item") else chosen[k])
              for k in ["objective", "learning_rate", "num_leaves", "min_child_samples", "feature_fraction",
                        "bagging_fraction", "lambda_l2"]}
    params["boosting"] = chosen_name
    cfg = {"params": params, "n_rounds": int(chosen.rounds), "valid_wape": float(chosen.valid_wape),
           "tuned_on": "condition A, validation = last 14 train days", "seed": SEED}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_config(MODELS_DIR / "lgb_config.json", cfg)

    L = ["# Step 6a report: LightGBM tuning (condition A only)\n",
         f"{a.trials} hyperparameter sets x 2 boosting types = {len(res)} fits; "
         f"{len(fit):,} fitting rows, {len(valid):,} validation rows. Took {time.time() - t0:.0f}s.\n",
         "## gbdt vs dart\n",
         "| boosting | best valid WAPE | median valid WAPE | mean seconds per fit |", "|---|---|---|---|"]
    for b in ["gbdt", "dart"]:
        s = res[res.boosting == b]
        L.append(f"| {b} | {s.valid_wape.min():.4f} | {s.valid_wape.median():.4f} | {s.seconds.mean():.0f} |")
    L += [f"\nPaired comparison (same hyperparameters): dart better in {dart_wins} of {a.trials} trials.",
          f"\n**Chosen: {chosen_name}** (valid WAPE {chosen.valid_wape:.4f}, {int(chosen.rounds)} rounds). "
          f"Tie rule: dart only if better than gbdt's best by > {TIE_MARGIN}.\n",
          "## Frozen configuration (used unchanged for A, B and C)\n", "```json", pd.Series(cfg).to_json(indent=2), "```",
          "\n## Top 10 trials\n", "```", res.sort_values("valid_wape").head(10).to_string(index=False), "```"]
    text = "\n".join(L)
    print(text)
    (PROJECT_ROOT / "outputs" / "step6a_tuning_report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
