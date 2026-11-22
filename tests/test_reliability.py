"""Unit tests for the Step 18 reliability warning: labels, threshold choice, metrics and the store split."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from blindspot.reliability import (DEPLOYED, FEATURES, REF_SEEDS, add_labels, best_f1_threshold, bootstrap_auc_ci,
                                   classification_report, fit_score, floor_threshold, make_methods,
                                   per_series_agreement, store_split)


def test_classification_report_known_values():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])
    r = classification_report(y, s, threshold=0.5)
    # Flags rows 0, 1, 3: TP 2, FP 1, FN 1, TN 4.
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 4)
    assert np.isclose(r["accuracy"], 6 / 8) and np.isclose(r["precision"], 2 / 3) and np.isclose(r["recall"], 2 / 3)
    assert np.isclose(r["specificity"], 4 / 5) and np.isclose(r["f1"], 2 / 3)
    assert np.isclose(r["roc_auc"], roc_auc_score(y, s)) and np.isclose(r["majority_accuracy"], 5 / 8)


def test_best_f1_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = best_f1_threshold(y, s)
    assert 0.3 < t <= 0.8 and classification_report(y, s, t)["f1"] == 1.0


def test_labels_use_noise_floor():
    items = pd.DataFrame({f"agree_ref{s}": np.linspace(0.5, 1.0, 100) for s in REF_SEEDS})
    thr = floor_threshold(items, q=0.05)
    assert np.isclose(thr, np.quantile(np.linspace(0.5, 1.0, 100), 0.05))
    for m in DEPLOYED:
        items[f"agree_{m}"] = [thr - 0.01, thr + 0.01] * 50
    lab = add_labels(items, thr)
    assert lab[f"unreliable_{DEPLOYED[0]}"].tolist() == [1, 0] * 50


def test_per_series_agreement_identical_is_one():
    rng = np.random.default_rng(0)
    sv = pd.DataFrame(rng.normal(size=(6, 5)))
    a = per_series_agreement(sv, sv, np.array([1, 1, 2, 2, 2, 3]))
    assert np.allclose(a, 1.0) and list(a.index) == [1, 2, 3]


def test_store_split_keeps_stores_together():
    items = pd.DataFrame({"store_id": np.repeat(np.arange(20), 5)})
    m = store_split(items)
    assert 0 < m.sum() < len(m)
    assert items[m].store_id.nunique() + items[~m].store_id.nunique() == 20
    assert not set(items[m].store_id) & set(items[~m].store_id)


def test_methods_learn_a_planted_signal():
    rng = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df["y"] = (df.model_disagreement + 0.3 * rng.normal(size=n) > 0.5).astype(int)
    tr, te = df.iloc[:400], df.iloc[400:]
    for name, method in make_methods().items():
        _, s = fit_score(method, tr, te, "y")
        auc = roc_auc_score(te.y, s)
        assert auc > (0.9 if name != "rule: stockout share" else 0.3), name


def test_bootstrap_auc_ci_contains_point():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    s = y + rng.normal(size=400)
    lo, hi = bootstrap_auc_ci(y, s, rng.integers(0, 40, 400))
    assert lo < roc_auc_score(y, s) < hi
