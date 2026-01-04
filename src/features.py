from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeatureSets:
    numeric: list[str]
    categorical: list[str]


def infer_feature_types(df: pd.DataFrame, target: str) -> FeatureSets:
    feature_cols = [c for c in df.columns if c != target]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feature_cols if c not in numeric]
    return FeatureSets(numeric=numeric, categorical=categorical)


def build_preprocessor(feature_sets: FeatureSets) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, feature_sets.numeric),
            ("cat", categorical_pipe, feature_sets.categorical),
        ]
    )
    return preprocessor
