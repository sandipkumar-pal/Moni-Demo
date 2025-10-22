from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency handled via requirements
    XGBClassifier = None

from .utils.metrics import build_confusion_matrix, compute_metrics
from .utils.preprocess import PreprocessResult, prepare_datasets


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    training_time: float


MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
}

if XGBClassifier is not None:
    MODEL_REGISTRY["XGBClassifier"] = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )


class UnsupportedModelError(ValueError):
    """Raised when a user requests an unsupported algorithm."""


def _instantiate_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise UnsupportedModelError(
            f"Model '{model_name}' is not supported. Available models: {list(MODEL_REGISTRY)}"
        )
    return clone(MODEL_REGISTRY[model_name])


@dataclass
class TrainingResponse:
    models: List[ModelResult]
    best_model: str
    metrics: Dict[str, Dict[str, float]]
    class_labels: List[str]
    feature_names: List[str]


def train_models(
    df: pd.DataFrame,
    target_column: str,
    selected_models: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResponse:
    """Train selected classification models and compute evaluation metrics."""
    if not selected_models:
        raise ValueError("At least one model must be selected for training.")

    preprocess_result: PreprocessResult = prepare_datasets(
        df, target_column, test_size=test_size, random_state=random_state
    )

    metrics_summary: Dict[str, Dict[str, float]] = {}
    model_results: List[ModelResult] = []

    for model_name in selected_models:
        estimator = _instantiate_model(model_name)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocess_result.preprocessor)),
                ("classifier", estimator),
            ]
        )

        start_time = time.time()
        pipeline.fit(preprocess_result.X_train, preprocess_result.y_train)
        training_time = time.time() - start_time

        y_pred = pipeline.predict(preprocess_result.X_test)

        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(preprocess_result.X_test)
            except Exception:  # pragma: no cover - fallback when predict_proba unavailable
                y_proba = None
        elif hasattr(pipeline, "decision_function"):
            try:
                decision_scores = pipeline.decision_function(preprocess_result.X_test)
                if decision_scores.ndim == 1:
                    y_proba = decision_scores
                else:
                    # Convert decision scores to probabilities via softmax approximation
                    exp_scores = np.exp(decision_scores)
                    y_proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            except Exception:  # pragma: no cover - fallback when decision function fails
                y_proba = None

        metrics = compute_metrics(
            preprocess_result.y_test,
            y_pred,
            y_proba,
        )
        metrics_summary[model_name] = metrics

        confusion = build_confusion_matrix(
            preprocess_result.y_test,
            y_pred,
            labels=np.arange(len(preprocess_result.class_labels)),
        )

        model_results.append(
            ModelResult(
                name=model_name,
                metrics=metrics,
                confusion_matrix=confusion.astype(int).tolist(),
                training_time=training_time,
            )
        )

    best_model = max(
        metrics_summary,
        key=lambda model: (
            metrics_summary[model].get("f1_score", float("nan")),
            metrics_summary[model].get("accuracy", float("nan")),
        ),
    )

    return TrainingResponse(
        models=model_results,
        best_model=best_model,
        metrics=metrics_summary,
        class_labels=preprocess_result.class_labels,
        feature_names=preprocess_result.feature_names,
    )


