"""Data preprocessing pipeline for crop recommendation models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

from utils.helpers import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    LABEL_ENCODER_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    ensure_directories,
    set_global_seed,
)


@dataclass
class PreprocessingArtifacts:
    """Fitted transformers used during model inference."""

    scaler: StandardScaler
    label_encoder: LabelEncoder


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the crop dataset from disk."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    return pd.read_csv(dataset_path)


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric feature nulls with median and drop rows with missing labels."""
    clean_df = df.copy()

    for col in FEATURE_COLUMNS:
        if col in clean_df.columns and clean_df[col].isna().any():
            clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    if clean_df[TARGET_COLUMN].isna().any():
        clean_df = clean_df.dropna(subset=[TARGET_COLUMN])

    return clean_df


def preprocess_data(
    dataset_path: Path = DATASET_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PreprocessingArtifacts]:
    """Prepare train/test data for model training.

    Steps:
    - load dataset
    - handle missing values
    - standard scale features
    - one-hot encode labels
    - split into train/test sets
    """
    set_global_seed(random_state)
    df = load_dataset(dataset_path)
    df = _handle_missing_values(df)

    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    X = df[FEATURE_COLUMNS].astype(float).values
    y_raw = df[TARGET_COLUMN].astype(str).values

    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y_raw)
    y_one_hot = to_categorical(y_int)

    X_train, X_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
        X,
        y_one_hot,
        y_int,
        test_size=test_size,
        random_state=random_state,
        stratify=y_int,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    artifacts = PreprocessingArtifacts(scaler=scaler, label_encoder=label_encoder)

    if save_artifacts:
        ensure_directories()
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    # Keep explicit cast for APIs expecting float32 tensors.
    return (
        X_train.astype(np.float32),
        X_test.astype(np.float32),
        y_train.astype(np.float32),
        y_test.astype(np.float32),
        artifacts,
    )


def preprocess_for_sklearn(
    dataset_path: Path = DATASET_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PreprocessingArtifacts]:
    """Prepare train/test data where labels are integer-encoded (for sklearn)."""
    X_train, X_test, y_train, y_test, artifacts = preprocess_data(
        dataset_path=dataset_path,
        test_size=test_size,
        random_state=random_state,
        save_artifacts=save_artifacts,
    )

    y_train_int = np.argmax(y_train, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
    return X_train, X_test, y_train_int, y_test_int, artifacts


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, artifacts = preprocess_data()
    print("Preprocessing complete")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Number of classes: {len(artifacts.label_encoder.classes_)}")
