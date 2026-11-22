"""Step 18: reliability warning - can we tell which items have a distorted explanation, without the true demand?

Binary classification per item (store x product). Label 1 = the deployed model's SHAP explanation for that item
disagrees with the clean model A more than retraining noise allows (Step 7 noise floor). Labels need model A, so
they exist only in this study; the warning itself uses only what a retailer has (stock records, promotions,
recorded sales, and models it can train on censored data).

Protocol (strictly out of sample):
  train  masking seed 42, items of a random half of the stores
  test   masking seed 43, items of the other half of the stores
The decision threshold of every warning is chosen on the training part (maximum F1) and then frozen.

Warnings compared: two single-feature rules (no training), logistic regression, gradient boosting.
Metrics on the test part: ROC curve and ROC AUC (95% CI over stores), PR AUC, accuracy (vs always predicting the
majority class), precision, recall, specificity, F1 and the confusion matrix.

Outputs: outputs/step18_reliability_report.md, step18_items.csv, step18_metrics.csv, step18_roc.png,
         step18_confusion.png
Usage:   .\\.venv\\Scripts\\python.exe scripts\\step18_reliability.py [--train-seed 42] [--test-seed 43]
         [--sample-size 5000]
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blindspot.data import PROJECT_ROOT  # noqa: E402
from blindspot.model import load_config  # noqa: E402
from blindspot.prepare import PROCESSED_DIR, nested_samples  # noqa: E402
from blindspot.reliability import (DEPLOYED, FEATURE_LABELS, FEATURES, FLOOR_Q, REF_SEEDS, add_labels,  # noqa: E402
                                   best_f1_threshold, bootstrap_auc_ci, classification_report, fit_score,
                                   floor_threshold, item_table, make_methods, roc_points, store_split)

OUT = PROJECT_ROOT / "outputs"
MODEL_LABELS = {"naive": "naive model (trained on recorded sales)", "censored_loss": "censored-loss model (Step 17)"}
lines: list[str] = []
problems: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def check(ok: bool, msg: str) -> None:
    out(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        problems.append(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--test-seed", type=int, default=43)
    ap.add_argument("--sample-size", type=int, default=5_000, help="smaller values are for quick trials")
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(OUT / "models" / "lgb_config.json")
    tab = pd.read_parquet(PROCESSED_DIR / "series.parquet")
    base = pd.read_parquet(PROCESSED_DIR / "sample_series.parquet")
    ids = nested_samples(tab, base, [a.sample_size])[a.sample_size].series_id
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet", filters=[("series_id", "in", ids.tolist())])
    panel = panel.sort_values(["series_id", "dt"]).reset_index(drop=True)

    out("# Step 18 report: reliability warning for explanations\n")
    out("Can a retailer tell which items have a distorted explanation, without knowing the true demand?\n")
    out("- **Unit:** one item (store x product), "
        f"{a.sample_size:,} items.")
    out(f"- **Label:** 1 = *unreliable*: the item's explanation (mean per-row SHAP rank agreement with the clean "
        f"model A over its clean test days) is below the {FLOOR_Q:.0%} quantile of the same agreement for A "
        f"retrained with other seeds ({', '.join(map(str, REF_SEEDS))}). In words: worse than retraining noise "
        f"produces for {1 - FLOOR_Q:.0%} of items.")
    out("- **Warning inputs** (no true demand needed): " + "; ".join(FEATURE_LABELS[f] for f in FEATURES) + ".")
    out(f"- **Out of sample:** train on masking seed {a.train_seed} and half the stores; test on masking seed "
        f"{a.test_seed} and the other stores. Thresholds are chosen on the training part (maximum F1).\n")

    # Item tables
    cache: dict = {}
    ts = time.time()
    tr_items = item_table(panel, a.train_seed, cfg, cache)
    print(f"  seed {a.train_seed}: {time.time() - ts:.0f}s", flush=True)
    ts = time.time()
    te_items = item_table(panel, a.test_seed, cfg, cache)
    print(f"  seed {a.test_seed}: {time.time() - ts:.0f}s", flush=True)
    thr = floor_threshold(tr_items)
    tr_items, te_items = add_labels(tr_items, thr), add_labels(te_items, thr)
    in_train = store_split(tr_items)
    train_stores = set(tr_items.store_id[in_train])
    train = tr_items[in_train].reset_index(drop=True)
    test = te_items[~te_items.store_id.isin(train_stores)].reset_index(drop=True)
    pd.concat([tr_items, te_items]).to_csv(OUT / "step18_items.csv", index=False)

    out("## Labels\n")
    ref_rate = np.mean([(tr_items[f"agree_ref{s}"] < thr).mean() for s in REF_SEEDS])
    out(f"Noise-floor threshold: per-item agreement < **{thr:.3f}**.\n")
    out("| model | train items | unreliable (train) | test items | unreliable (test) | mean agreement with A (test) |")
    out("|---|---|---|---|---|---|")
    out(f"| A retrained (A', reference) | {len(tr_items):,} | {ref_rate:.1%} | - | - | "
        f"{np.mean([te_items[f'agree_ref{s}'].mean() for s in REF_SEEDS]):.3f} |")
    for m in DEPLOYED:
        out(f"| {MODEL_LABELS[m]} | {len(train):,} | {train[f'unreliable_{m}'].mean():.1%} | {len(test):,} | "
            f"{test[f'unreliable_{m}'].mean():.1%} | {test[f'agree_{m}'].mean():.3f} |")
    out("")
    check(abs(ref_rate - FLOOR_Q) < 0.01, f"about {FLOOR_Q:.0%} of A' items fall below the threshold (by construction)")
    check(not set(test.store_id) & train_stores, "no store is in both the training and the test part")
    for m in DEPLOYED:
        check(train[f"unreliable_{m}"].sum() >= 20 and test[f"unreliable_{m}"].sum() >= 20,
              f"{m}: at least 20 unreliable items in each part (enough to evaluate)")

    # Evaluate warnings
    rows, roc, best = [], {}, {}
    for m in DEPLOYED:
        label = f"unreliable_{m}"
        y_tr, y_te = train[label].to_numpy(), test[label].to_numpy()
        roc[m] = {}
        for name, method in make_methods().items():
            s_tr, s_te = fit_score(method, train, test, label)
            r = classification_report(y_te, s_te, best_f1_threshold(y_tr, s_tr))
            r["auc_lo"], r["auc_hi"] = bootstrap_auc_ci(y_te, s_te, test.store_id.to_numpy())
            rows.append({"model": m, "method": name, **r})
            roc[m][name] = roc_points(y_te, s_te)
            if name == "logistic regression":
                coef = pd.Series(method[-1].coef_[0], index=FEATURES)
                roc[m]["_coef"] = coef
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "step18_metrics.csv", index=False)

    for m in DEPLOYED:
        r = res[res.model == m].set_index("method")
        best[m] = r.roc_auc.idxmax()
        out(f"\n## Warning for the {MODEL_LABELS[m]}\n")
        out(f"Test items: {int(r.n.iloc[0]):,}, unreliable: {int(r.positives.iloc[0]):,} "
            f"({r.prevalence.iloc[0]:.1%}). Always answering \"reliable\" would score accuracy "
            f"{r.majority_accuracy.iloc[0]:.1%}, so accuracy alone is not informative here.\n")
        out("| warning | ROC AUC (95% CI) | PR AUC | accuracy | precision | recall | specificity | F1 |")
        out("|---|---|---|---|---|---|---|---|")
        for name, x in r.iterrows():
            out(f"| {name} | {x.roc_auc:.3f} ({x.auc_lo:.3f}-{x.auc_hi:.3f}) | {x.pr_auc:.3f} | {x.accuracy:.1%} | "
                f"{x.precision:.1%} | {x.recall:.1%} | {x.specificity:.1%} | {x.f1:.3f} |")
        out(f"\nPR AUC of a random warning = prevalence = {r.prevalence.iloc[0]:.3f}; ROC AUC of a random warning = 0.5.\n")
        out("**Confusion matrices** (test items; rows = truth, columns = warning):\n")
        out("| warning | true reliable, no warning (TN) | false alarm (FP) | missed (FN) | caught (TP) |")
        out("|---|---|---|---|---|")
        for name, x in r.iterrows():
            out(f"| {name} | {x.tn:,} | {x.fp:,} | {x.fn:,} | {x.tp:,} |")
        flat = [n for n, x in r.iterrows() if x.specificity == 0 or x.recall == 0]
        if flat:
            out(f"\n{', '.join(flat)}: the best training threshold flags every item (or none), i.e. the score cannot "
                "separate the classes at any useful cut-off; judge it by ROC AUC, not by its threshold metrics.")
        coef = roc[m]["_coef"].sort_values(key=abs, ascending=False)
        out("\nLogistic regression weights (standardised inputs; positive = more likely unreliable):\n")
        out("| input | weight |")
        out("|---|---|")
        for f, w in coef.items():
            out(f"| {FEATURE_LABELS[f]} | {w:+.2f} |")
        x = r.loc[best[m]]
        out(f"\n- Best warning by ROC AUC: **{best[m]}** (AUC {x.roc_auc:.3f}, CI {x.auc_lo:.3f}-{x.auc_hi:.3f}). "
            + ("The CI excludes 0.5: the warning ranks items better than chance." if x.auc_lo > 0.5 else
               "The CI includes 0.5: the warning is not reliably better than chance."))
        out(f"- At its frozen threshold it flags {x.tp + x.fp:,} of {int(x.n):,} items and catches {x.tp:,} of "
            f"{x.tp + x.fn:,} unreliable ones (recall {x.recall:.0%}); {x.precision:.0%} of its warnings are right.")

    # Plots
    fig, axs = plt.subplots(1, len(DEPLOYED), figsize=(6 * len(DEPLOYED), 4.6))
    for ax, m in zip(np.atleast_1d(axs), DEPLOYED):
        for name, (fpr, tpr) in ((k, v) for k, v in roc[m].items() if not k.startswith("_")):
            auc = res[(res.model == m) & (res.method == name)].roc_auc.iloc[0]
            ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.2f})")
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8, label="chance (AUC 0.50)")
        ax.set_title(MODEL_LABELS[m], fontsize=10)
        ax.set_xlabel("false positive rate (reliable items flagged)")
        ax.set_ylabel("true positive rate (unreliable items caught)")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "step18_roc.png", dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(1, len(DEPLOYED), figsize=(5 * len(DEPLOYED), 4.2))
    for ax, m in zip(np.atleast_1d(axs), DEPLOYED):
        x = res[(res.model == m) & (res.method == best[m])].iloc[0]
        cm = np.array([[x.tn, x.fp], [x.fn, x.tp]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm.sum():.0%})", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["no warning", "warning"])
        ax.set_yticks([0, 1], ["reliable", "unreliable"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("truth")
        ax.set_title(f"{MODEL_LABELS[m]}\n{best[m]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "step18_confusion.png", dpi=110)
    plt.close(fig)
    out("\n![ROC curves](step18_roc.png)\n\n![confusion matrices](step18_confusion.png)")

    out(f"\n## Summary\n\n{'ALL CHECKS PASSED' if not problems else 'FAILED CHECKS:'}")
    for p in problems:
        out(f"- {p}")
    out(f"\nDone in {time.time() - t0:.0f}s.")
    (OUT / "step18_reliability_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {OUT / 'step18_reliability_report.md'}")


if __name__ == "__main__":
    main()
