"""Ensemble inference by averaging ANN and Random Forest probabilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.helpers import (
    ANN_MODEL_PATH,
    FEATURE_COLUMNS,
    LABEL_ENCODER_PATH,
    RF_MODEL_PATH,
    SCALER_PATH,
    PredictionResult,
    prepare_feature_frame,
)


class EnsemblePredictor:
    """Wrap ANN + RF models and expose a single prediction interface."""

    def __init__(self) -> None:
        self.ann_model = self._load_ann_model()
        self.rf_model = self._load_rf_model()
        self.scaler = self._load_scaler()
        self.label_encoder = self._load_label_encoder()

    @staticmethod
    def _load_ann_model() -> tf.keras.Model:
        if not ANN_MODEL_PATH.exists():
            raise FileNotFoundError(f"ANN model not found: {ANN_MODEL_PATH}")
        return tf.keras.models.load_model(ANN_MODEL_PATH)

    @staticmethod
    def _load_rf_model():
        if not RF_MODEL_PATH.exists():
            raise FileNotFoundError(f"RF model not found: {RF_MODEL_PATH}")
        return joblib.load(RF_MODEL_PATH)

    @staticmethod
    def _load_scaler():
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
        return joblib.load(SCALER_PATH)

    @staticmethod
    def _load_label_encoder():
        if not LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError(f"Label encoder not found: {LABEL_ENCODER_PATH}")
        return joblib.load(LABEL_ENCODER_PATH)

    def _ensure_frame(self, payload: Union[Dict[str, float], pd.DataFrame]) -> pd.DataFrame:
        if isinstance(payload, pd.DataFrame):
            return payload[FEATURE_COLUMNS].astype(float)
        return prepare_feature_frame(payload)

    def predict_proba(self, payload: Union[Dict[str, float], pd.DataFrame]) -> np.ndarray:
        """Return averaged class probabilities from ANN and RF models."""
        frame = self._ensure_frame(payload)
        X_scaled = self.scaler.transform(frame.values)

        ann_probs = self.ann_model.predict(X_scaled, verbose=0)
        rf_probs = self.rf_model.predict_proba(X_scaled)

        if ann_probs.shape != rf_probs.shape:
            raise ValueError(
                f"Model probability shape mismatch: ANN={ann_probs.shape}, RF={rf_probs.shape}"
            )

        return (ann_probs + rf_probs) / 2.0

    def predict(self, payload: Union[Dict[str, float], pd.DataFrame]) -> PredictionResult:
        """Return recommended crop, confidence, and class distribution."""
        ensemble_probs = self.predict_proba(payload)
        class_idx = int(np.argmax(ensemble_probs[0]))
        confidence = float(ensemble_probs[0, class_idx])
        crop = str(self.label_encoder.inverse_transform([class_idx])[0])

        probabilities = {
            str(c): float(p)
            for c, p in zip(self.label_encoder.classes_, ensemble_probs[0])
        }

        return PredictionResult(
            recommended_crop=crop,
            confidence=confidence,
            probabilities=probabilities,
        )


if __name__ == "__main__":
    sample = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 25,
        "humidity": 80,
        "ph": 6.5,
        "rainfall": 200,
    }
    predictor = EnsemblePredictor()
    result = predictor.predict(sample)
    print(result)
