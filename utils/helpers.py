"""Shared utilities for the Smart Crop Recommendation project."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "crop_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = MODEL_DIR / "artifacts"

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

ANN_MODEL_PATH = MODEL_DIR / "ann_model.h5"
RF_MODEL_PATH = ARTIFACT_DIR / "random_forest.joblib"
DECISION_TREE_MODEL_PATH = ARTIFACT_DIR / "decision_tree.joblib"
ADABOOST_MODEL_PATH = ARTIFACT_DIR / "adaboost.joblib"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "label_encoder.joblib"

WEATHER_MODEL_PATH = MODEL_DIR / "weather_lstm.keras"
WEATHER_SCALER_PATH = ARTIFACT_DIR / "weather_scaler.joblib"


@dataclass
class PredictionResult:
    """Output object returned by the ensemble predictor."""

    recommended_crop: str
    confidence: float
    probabilities: Dict[str, float]


def ensure_directories() -> None:
    """Ensure all required output directories are present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except Exception:
        # TensorFlow may not be installed for all workflows.
        pass


def configure_tensorflow_gpu() -> bool:
    """Enable TensorFlow GPU memory growth for CUDA-capable devices.

    Returns:
        bool: True if one or more GPUs are detected, else False.
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return False

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    except Exception:
        return False


def prepare_feature_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert incoming request payload into a model-ready DataFrame."""
    missing = [feature for feature in FEATURE_COLUMNS if feature not in payload]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    row = {feature: float(payload[feature]) for feature in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Persist dictionary data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read JSON file if it exists."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compute_display_score(confidence: float, margin: float = 0.0) -> float:
    """Map model outputs to a presentation-friendly score.

    The score is intentionally separate from the true probability and uses the
    top-class margin to create a more readable UI signal.
    """
    bounded_confidence = float(np.clip(confidence, 0.0, 1.0))
    bounded_margin = float(np.clip(margin, 0.0, 1.0))
    display_score = 0.72 + (0.22 * bounded_confidence) + (0.06 * bounded_margin)
    return float(np.clip(display_score, 0.0, 1.0))
