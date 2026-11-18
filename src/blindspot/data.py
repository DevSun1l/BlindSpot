"""Loading FreshRetailNet-50K and caching it locally as parquet.

Conventions taken from frn-50k-baseline (Apache 2.0, Dingdong-Inc):
- train split: 90 days per store-product series, ending 2024-06-25
- eval split: the 7 following days (2024-06-26 onward)
- hours_stock_status == 1 means out of stock; business hours are 6:00-22:00 (hour indices 6..21)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_ID = "Dingdong-Inc/FreshRetailNet-50K"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

KEYS = ["store_id", "product_id", "dt"]
EXPECTED_COLUMNS = [
    "city_id", "store_id", "management_group_id", "first_category_id",
    "second_category_id", "third_category_id", "product_id", "dt",
    "sale_amount", "hours_sale", "stock_hour6_22_cnt", "hours_stock_status",
    "discount", "holiday_flag", "activity_flag", "precpt",
    "avg_temperature", "avg_humidity", "avg_wind_level",
]
TRAIN_DAYS = 90
EVAL_DAYS = 7
BUSINESS_HOURS = slice(6, 22)  # hour indices 6..21 -> 16 hours, matches stock_hour6_22_cnt
OUT_OF_STOCK = 1


def load_split(split: str, refresh: bool = False) -> pd.DataFrame:
    """Return the 'train' or 'eval' split as a DataFrame, cached at data/raw/<split>.parquet."""
    if split not in ("train", "eval"):
        raise ValueError("split must be 'train' or 'eval'")
    path = RAW_DIR / f"{split}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    from datasets import load_dataset  # imported lazily: slow import

    df = load_dataset(DATASET_ID, split=split).to_pandas()
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(KEYS).reset_index(drop=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
