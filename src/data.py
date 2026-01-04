from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class DatasetConfig:
    target: str
    positive_label: str
    id_cols: list[str]
    optional_time_col: Optional[str]


def load_raw_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def prepare_dataset(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    df = clean_telco(df)
    drop_cols = [c for c in cfg.id_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df
