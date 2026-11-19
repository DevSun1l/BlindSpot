"""LightGBM training, validation, and prediction utilities for conditions A, B, and C.

Hyperparameters are tuned on condition A's validation split and reused unchanged
across all three conditions so model differences reflect training data rather
than configuration changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from blindspot.features import CATEGORICAL, FEATURES
from blindspot.metrics import evaluate
from blindspot.prepare import PROCESSED_DIR

SEED = 42
MAX_ROUNDS_GBDT = 3000
EARLY_STOP = 100


def load_features(condition: str) -> pd.DataFrame:
    """Load the prebuilt feature table for condition 'A', 'B', or 'C'."""
    return pd.read_parquet(PROCESSED_DIR / f"features_{condition}.parquet")


def split_rows(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a feature table into (fit, valid, all_train, test) subsets."""
    tt = f.is_train_target == 1
    return f[tt & (f.is_valid == 0)], f[f.is_valid == 1], f[tt], f[f.is_test == 1]


def base_params(p: dict) -> dict:
    """Merge user hyperparameters with deterministic LightGBM defaults."""
    out = {"verbosity": -1, "seed": SEED, "deterministic": True, "force_row_wise": True, "num_threads": 0,
           "bagging_freq": 1, **p}
    if out.get("objective") == "tweedie":
        out.setdefault("tweedie_variance_power", 1.3)
    return out


def _dataset(df: pd.DataFrame, ref: lgb.Dataset | None = None, features: list[str] | None = None) -> lgb.Dataset:
    return lgb.Dataset(df[features or FEATURES], label=df["target"], categorical_feature=CATEGORICAL, reference=ref,
                       free_raw_data=False)


def fit_and_validate(params: dict, fit: pd.DataFrame, valid: pd.DataFrame) -> tuple[float, int]:
    """Train on `fit` and return (validation WAPE, n_rounds).

    GBDT uses early stopping on `valid`; DART trains for the fixed `n_estimators`
    count because LightGBM does not support early stopping with DART.
    """
    p = base_params({k: v for k, v in params.items() if k != "n_estimators"})
    dfit = _dataset(fit)
    dval = _dataset(valid, dfit)
    if p["boosting"] == "gbdt":
        booster = lgb.train(p, dfit, num_boost_round=MAX_ROUNDS_GBDT, valid_sets=[dval],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        n = booster.best_iteration
    else:
        n = int(params["n_estimators"])
        booster = lgb.train(p, dfit, num_boost_round=n)
    v = valid.assign(pred=np.clip(booster.predict(valid[FEATURES], num_iteration=n), 0, None))
    wape = float(evaluate(v, "pred", y="target").set_index("group").loc["all", "wape"])
    return wape, n


def train_final(params: dict, n_rounds: int, train: pd.DataFrame, features: list[str] | None = None) -> lgb.Booster:
    """Fit a LightGBM booster on `train` for a fixed number of boosting rounds."""
    p = base_params({k: v for k, v in params.items() if k != "n_estimators"})
    return lgb.train(p, _dataset(train, features=features), num_boost_round=n_rounds)


def predict(booster: lgb.Booster, df: pd.DataFrame, features: list[str] | None = None) -> np.ndarray:
    """Generate non-negative demand forecasts."""
    return np.clip(booster.predict(df[features or FEATURES]), 0, None)


def save_config(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg, indent=2))


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())
