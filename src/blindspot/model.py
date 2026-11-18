"""Step 6: LightGBM training for conditions A, B and C.

Hyperparameters (including gbdt vs dart) are tuned ONCE on condition A's validation days, then
frozen and reused unchanged for B and C, so any difference between the models comes from the
training data, not the settings.
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
    return pd.read_parquet(PROCESSED_DIR / f"features_{condition}.parquet")


def split_rows(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(fit, valid, all_train, test) row subsets."""
    tt = f.is_train_target == 1
    return f[tt & (f.is_valid == 0)], f[f.is_valid == 1], f[tt], f[f.is_test == 1]


def base_params(p: dict) -> dict:
    out = {"verbosity": -1, "seed": SEED, "deterministic": True, "force_row_wise": True, "num_threads": 0,
           "bagging_freq": 1, **p}
    if out.get("objective") == "tweedie":
        out.setdefault("tweedie_variance_power", 1.3)
    return out


def _dataset(df: pd.DataFrame, ref: lgb.Dataset | None = None) -> lgb.Dataset:
    return lgb.Dataset(df[FEATURES], label=df["target"], categorical_feature=CATEGORICAL, reference=ref,
                       free_raw_data=False)


def fit_and_validate(params: dict, fit: pd.DataFrame, valid: pd.DataFrame) -> tuple[float, int]:
    """Train on `fit`, score WAPE on `valid` (baseline protocol). Returns (valid_wape, n_rounds).
    gbdt: early stopping picks n_rounds. dart: n_rounds = params['n_estimators'] (no early stopping)."""
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


def train_final(params: dict, n_rounds: int, train: pd.DataFrame) -> lgb.Booster:
    p = base_params({k: v for k, v in params.items() if k != "n_estimators"})
    return lgb.train(p, _dataset(train), num_boost_round=n_rounds)


def predict(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    return np.clip(booster.predict(df[FEATURES]), 0, None)


def save_config(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg, indent=2))


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())
