"""Ensemble inference by averaging AdaBoost and Decision Tree probabilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from preprocessing.preprocess import preprocess_for_sklearn
from utils.helpers import (
    ADABOOST_MODEL_PATH,
    DECISION_TREE_MODEL_PATH,
    FEATURE_COLUMNS,
    LABEL_ENCODER_PATH,
    SCALER_PATH,
    PredictionResult,
    ensure_directories,
    prepare_feature_frame,
    set_global_seed,
)

ADABOOST_WEIGHT = 0.85
DECISION_TREE_WEIGHT = 0.15
PROBABILITY_SHARPNESS = 1.0


class EnsemblePredictor:
    """Wrap AdaBoost + Decision Tree models and expose a single prediction interface."""

    def __init__(self) -> None:
        self.adaboost_model = self._load_adaboost_model()
        self.decision_tree_model = self._load_decision_tree_model()
        self.scaler = self._load_scaler()
        self.label_encoder = self._load_label_encoder()

    @staticmethod
    def _load_adaboost_model() -> AdaBoostClassifier:
        if not ADABOOST_MODEL_PATH.exists():
            raise FileNotFoundError(f"AdaBoost model not found: {ADABOOST_MODEL_PATH}")
        return joblib.load(ADABOOST_MODEL_PATH)

    @staticmethod
    def _load_decision_tree_model() -> DecisionTreeClassifier:
        if not DECISION_TREE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Decision Tree model not found: {DECISION_TREE_MODEL_PATH}")
        return joblib.load(DECISION_TREE_MODEL_PATH)

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
        """Return sharpened ensemble probabilities from AdaBoost and Decision Tree models."""
        frame = self._ensure_frame(payload)
        X_scaled = self.scaler.transform(frame.values)

        adaboost_probs = self.adaboost_model.predict_proba(X_scaled)
        tree_probs = self.decision_tree_model.predict_proba(X_scaled)

        if adaboost_probs.shape != tree_probs.shape:
            raise ValueError(
                f"Model probability shape mismatch: AdaBoost={adaboost_probs.shape}, DecisionTree={tree_probs.shape}"
            )

        combined_probs = (ADABOOST_WEIGHT * adaboost_probs) + (DECISION_TREE_WEIGHT * tree_probs)
        if PROBABILITY_SHARPNESS != 1.0:
            combined_probs = np.power(np.clip(combined_probs, 1e-12, 1.0), PROBABILITY_SHARPNESS)
            combined_probs /= combined_probs.sum(axis=1, keepdims=True)
        return combined_probs / combined_probs.sum(axis=1, keepdims=True)

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


def _build_adaboost(random_state: int) -> AdaBoostClassifier:
    """Create an AdaBoost classifier with a shallow decision tree base learner."""
    base_tree = DecisionTreeClassifier(max_depth=3, random_state=random_state)
    try:
        return AdaBoostClassifier(
            estimator=base_tree,
            n_estimators=200,
            learning_rate=0.5,
            random_state=random_state,
        )
    except TypeError:
        return AdaBoostClassifier(
            base_estimator=base_tree,
            n_estimators=200,
            learning_rate=0.5,
            random_state=random_state,
        )


def train_ensemble_models(random_state: int = 42) -> Tuple[Dict[str, object], Dict[str, float]]:
    """Train and persist AdaBoost + Decision Tree models for crop prediction."""
    set_global_seed(random_state)
    ensure_directories()

    X_train, X_test, y_train, y_test, _ = preprocess_for_sklearn(
        random_state=random_state,
        save_artifacts=True,
        augment_train_data=True,
        augmentation_factor=0.5,
        augmentation_noise=0.03,
    )

    decision_tree = DecisionTreeClassifier(random_state=random_state)
    adaboost = _build_adaboost(random_state=random_state)

    decision_tree.fit(X_train, y_train)
    adaboost.fit(X_train, y_train)

    tree_probs = decision_tree.predict_proba(X_test)
    adaboost_probs = adaboost.predict_proba(X_test)
    ensemble_probs = (tree_probs + adaboost_probs) / 2.0

    tree_pred = decision_tree.predict(X_test)
    adaboost_pred = adaboost.predict(X_test)
    ensemble_pred = np.argmax(ensemble_probs, axis=1)

    tree_accuracy = accuracy_score(y_test, tree_pred)
    adaboost_accuracy = accuracy_score(y_test, adaboost_pred)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

    joblib.dump(decision_tree, DECISION_TREE_MODEL_PATH)
    joblib.dump(adaboost, ADABOOST_MODEL_PATH)

    print(f"Decision Tree test accuracy: {tree_accuracy:.4f}")
    print(f"AdaBoost test accuracy: {adaboost_accuracy:.4f}")
    print(f"Ensemble test accuracy: {ensemble_accuracy:.4f}")
    print(f"Decision Tree model saved at: {DECISION_TREE_MODEL_PATH}")
    print(f"AdaBoost model saved at: {ADABOOST_MODEL_PATH}")
    print("Decision Tree classification report:")
    print(classification_report(y_test, tree_pred))
    print("AdaBoost classification report:")
    print(classification_report(y_test, adaboost_pred))

    metrics = {
        "decision_tree_accuracy": float(tree_accuracy),
        "adaboost_accuracy": float(adaboost_accuracy),
        "ensemble_accuracy": float(ensemble_accuracy),
        "n_estimators": 200.0,
        "adaboost_weight": ADABOOST_WEIGHT,
        "decision_tree_weight": DECISION_TREE_WEIGHT,
        "probability_sharpness": PROBABILITY_SHARPNESS,
    }
    artifacts = {
        "decision_tree": decision_tree,
        "adaboost": adaboost,
    }
    return artifacts, metrics


if __name__ == "__main__":
    train_ensemble_models()
