"""Train and save the Random Forest model for crop recommendation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from preprocessing.preprocess import preprocess_for_sklearn
from utils.helpers import RF_MODEL_PATH, ensure_directories, set_global_seed


def train_random_forest(random_state: int = 42) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Train and persist RandomForestClassifier using project defaults."""
    set_global_seed(random_state)
    ensure_directories()

    X_train, X_test, y_train, y_test, _ = preprocess_for_sklearn(
        random_state=random_state,
        save_artifacts=True,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, RF_MODEL_PATH)

    print(f"RF test accuracy: {test_accuracy:.4f}")
    print(f"RF model saved at: {RF_MODEL_PATH}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    metrics = {
        "test_accuracy": float(test_accuracy),
        "n_estimators": 200.0,
    }
    return model, metrics


if __name__ == "__main__":
    train_random_forest()
