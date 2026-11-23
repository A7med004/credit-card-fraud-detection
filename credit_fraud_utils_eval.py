"""Evaluation, threshold selection, and artifact helpers."""

from __future__ import annotations

from pathlib import Path
import json
import pickle
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Select the threshold with the highest F1 score on held-out data."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_values = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_index = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[best_index]), float(f1_values[best_index])


def evaluate_probabilities(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate threshold-dependent and ranking metrics for fraud probabilities."""
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "threshold": float(threshold),
        "predicted_fraud_count": int(predictions.sum()),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def save_model_artifact(model: Any, threshold: float, model_name: str, feature_columns: list[str], output_path: str | Path) -> None:
    """Save the required model dictionary in a portable pickle artifact."""
    artifact = {
        "model": model,
        "threshold": float(threshold),
        "model_name": model_name,
        "feature_columns": feature_columns,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(artifact, file)


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Write human-readable validation metrics beside the model artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
