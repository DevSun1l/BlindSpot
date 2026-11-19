"""Load FreshRetailNet-50K splits from Hugging Face and cache them as local Parquet files.

Dataset conventions (following frn-50k-baseline):
- train split: 90 days per store-product series, ending 2024-06-25
- eval split: 7 subsequent days starting 2024-06-26
- hours_stock_status == 1 indicates out of stock; business hours are 6:00-22:00 (indices 6..21)
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
BUSINESS_HOURS = slice(6, 22)  # 16 business hours (06:00-21:59), aligned with stock_hour6_22_cnt
OUT_OF_STOCK = 1


def load_split(split: str, refresh: bool = False) -> pd.DataFrame:
    """Return the 'train' or 'eval' split as a DataFrame, reading from local Parquet cache when available."""
    if split not in ("train", "eval"):
        raise ValueError("split must be 'train' or 'eval'")
    path = RAW_DIR / f"{split}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    # Defer Hugging Face datasets import so cached reads stay fast.
    from datasets import load_dataset

    df = load_dataset(DATASET_ID, split=split).to_pandas()
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(KEYS).reset_index(drop=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
