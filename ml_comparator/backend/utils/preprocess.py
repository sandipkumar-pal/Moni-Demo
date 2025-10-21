from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


@dataclass
class PreprocessResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    label_encoder: LabelEncoder
    feature_names: List[str]
    class_labels: List[str]


def identify_feature_types(features: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical feature names."""
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = (
        features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    )
    numeric_features = [col for col in numeric_features if col not in categorical_features]
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """Create a column transformer for numeric and categorical features."""
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        # When all columns are the target or dataset is empty
        raise ValueError("No features available for preprocessing.")

    return ColumnTransformer(transformers)


def prepare_datasets(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessResult:
    """Split the dataset and return preprocessing metadata."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    if df[target_column].nunique() < 2:
        raise ValueError("Target column must contain at least two distinct classes.")

    features = df.drop(columns=[target_column])
    labels = df[target_column]

    numeric_features, categorical_features = identify_feature_types(features)
    feature_names = numeric_features + categorical_features

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    stratify = encoded_labels if len(np.unique(encoded_labels)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        encoded_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        feature_names=feature_names,
        class_labels=label_encoder.classes_.tolist(),
    )


