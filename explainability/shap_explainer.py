"""SHAP explainability for crop recommendation predictions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.helpers import ADABOOST_MODEL_PATH, DECISION_TREE_MODEL_PATH, FEATURE_COLUMNS, SCALER_PATH, prepare_feature_frame


class CropShapExplainer:
    """Explain crop predictions using the Decision Tree member of the ensemble."""

    def __init__(self) -> None:
        if not SCALER_PATH.exists():
            raise FileNotFoundError("Scaler artifact is missing. Train models first.")

        self.tree_model = self._load_tree_model()
        self.scaler = joblib.load(SCALER_PATH)
        try:
            self.explainer = shap.TreeExplainer(self.tree_model)
        except Exception:
            if ADABOOST_MODEL_PATH.exists() and DECISION_TREE_MODEL_PATH.exists():
                self.tree_model = joblib.load(DECISION_TREE_MODEL_PATH)
                self.explainer = shap.TreeExplainer(self.tree_model)
            else:
                raise

    @staticmethod
    def _load_tree_model() -> object:
        """Load the preferred tree ensemble artifact for SHAP explanations."""
        if DECISION_TREE_MODEL_PATH.exists():
            return joblib.load(DECISION_TREE_MODEL_PATH)
        if ADABOOST_MODEL_PATH.exists():
            return joblib.load(ADABOOST_MODEL_PATH)
        raise FileNotFoundError(
            "AdaBoost or Decision Tree model artifacts are missing. Train models first."
        )

    def _extract_class_shap(self, shap_values, class_idx: int) -> np.ndarray:
        """Handle SHAP output format differences across versions."""
        if isinstance(shap_values, list):
            return np.array(shap_values[class_idx])[0]

        values = np.array(shap_values)
        if values.ndim == 3:
            # shape: (n_samples, n_features, n_classes)
            return values[0, :, class_idx]
        if values.ndim == 2:
            # Binary fallback shape
            return values[0]
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}")

    def explain_top_features(
        self,
        payload: Dict[str, float],
        predicted_class_index: int,
        top_k: int = 3,
    ) -> Dict[str, float]:
        """Return top contributing features for the predicted class."""
        frame = prepare_feature_frame(payload)
        X_scaled = self.scaler.transform(frame[FEATURE_COLUMNS].values)

        shap_values = self.explainer.shap_values(X_scaled)
        contributions = self._extract_class_shap(shap_values, predicted_class_index)

        pairs = list(zip(FEATURE_COLUMNS, contributions.tolist()))
        pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        top_pairs = pairs[:top_k]
        return {feature: float(value) for feature, value in top_pairs}

    def plot_explanation(self, explanation: Dict[str, float]) -> plt.Figure:
        """Create horizontal bar chart for feature contributions."""
        features = list(explanation.keys())
        values = [explanation[f] for f in features]

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#1f77b4" if v >= 0 else "#d62728" for v in values]
        ax.barh(features, values, color=colors)
        ax.set_title("Top SHAP Feature Contributions")
        ax.set_xlabel("Contribution Value")
        ax.invert_yaxis()
        plt.tight_layout()
        return fig
