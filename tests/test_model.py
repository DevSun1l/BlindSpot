"""Unit tests for LightGBM training helpers and validation of the saved tuning config."""
import numpy as np
import pandas as pd
import pytest

from blindspot.data import PROJECT_ROOT
from blindspot.features import CATEGORICAL, FEATURES
from blindspot.model import fit_and_validate, load_config, predict, train_final


def _toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    for c in CATEGORICAL:
        df[c] = pd.Categorical(rng.integers(0, 5, n))
    df["target"] = np.clip(2 + df["lag_1"] + 0.5 * df["discount"] + rng.normal(0, 0.1, n), 0, None)
    df["dt"] = np.repeat(np.arange(n // 100), 100)
    df["sales_group"] = "high"
    df["is_clean_day"] = 1
    return df


PARAMS = {"objective": "regression", "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 20,
          "feature_fraction": 1.0, "bagging_fraction": 1.0, "lambda_l2": 0.0, "n_estimators": 50}


@pytest.mark.parametrize("boosting", ["gbdt", "dart"])
def test_fit_and_validate_learns(boosting):
    df = _toy()
    wape, n = fit_and_validate({**PARAMS, "boosting": boosting}, df.iloc[:2400], df.iloc[2400:])
    assert wape < 0.2 and n > 0


def test_training_is_deterministic():
    df = _toy()
    p = {**PARAMS, "boosting": "gbdt"}
    a, b = train_final(p, 30, df), train_final(p, 30, df)
    assert np.array_equal(predict(a, df), predict(b, df))


cfg_path = PROJECT_ROOT / "outputs" / "models" / "lgb_config.json"


@pytest.mark.skipif(not cfg_path.exists(), reason="run scripts/step6a_tune.py first")
def test_frozen_config_is_complete():
    cfg = load_config(cfg_path)
    assert cfg["params"]["boosting"] in {"gbdt", "dart"} and cfg["n_rounds"] > 0
