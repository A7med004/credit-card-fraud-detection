"""Data loading and validation helpers for the fraud-detection project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = {"Time", "Amount", TARGET_COLUMN, *[f"V{i}" for i in range(1, 29)]}


def load_dataset(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    """Load a CSV file and fail early when its schema is not the expected one."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset was not found: {path}")

    frame = pd.read_csv(path)
    expected = REQUIRED_COLUMNS if require_target else REQUIRED_COLUMNS - {TARGET_COLUMN}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values; imputation is not configured.")
    if require_target and not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only binary labels: 0 and 1.")
    return frame


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model features in their original, deterministic order."""
    return [column for column in frame.columns if column != TARGET_COLUMN]


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features and the binary target."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}'.")
    return frame[get_feature_columns(frame)], frame[TARGET_COLUMN].astype(int)


def combine_datasets(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load, concatenate, and de-duplicate multiple labelled dataset files."""
    frames = [load_dataset(path) for path in paths]
    if not frames:
        raise ValueError("At least one dataset path is required.")
    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
