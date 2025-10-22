from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, F1, and ROC-AUC metrics."""
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    roc_auc_value: Optional[float] = None

    if y_proba is not None:
        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                roc_auc_value = float(roc_auc_score(y_true, y_proba))
            else:
                classes = np.unique(y_true)
                y_true_binarized = label_binarize(y_true, classes=classes)
                roc_auc_value = float(
                    roc_auc_score(
                        y_true_binarized,
                        y_proba,
                        average=average,
                        multi_class="ovr",
                    )
                )
        except ValueError:
            roc_auc_value = None

    results["roc_auc"] = roc_auc_value

    return results


def build_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return a confusion matrix for the predictions."""
    return confusion_matrix(y_true, y_pred, labels=labels)


