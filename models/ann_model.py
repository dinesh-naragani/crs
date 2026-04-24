"""Train and save the ANN model for crop recommendation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from preprocessing.preprocess import preprocess_data
from utils.helpers import ANN_MODEL_PATH, configure_tensorflow_gpu, ensure_directories, set_global_seed


def build_ann_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    """Construct ANN architecture defined in the requirements."""
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.1),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_ann_model(
    epochs: int = 120,
    batch_size: int = 32,
    random_state: int = 42,
) -> Tuple[tf.keras.Model, Dict[str, float]]:
    """Train ANN model and persist as models/ann_model.h5."""
    set_global_seed(random_state)
    ensure_directories()
    gpu_available = configure_tensorflow_gpu()

    X_train, X_test, y_train, y_test, artifacts = preprocess_data(
        random_state=random_state,
        save_artifacts=True,
        augment_train_data=True,
        augmentation_factor=0.5,
        augmentation_noise=0.03,
    )

    model = build_ann_model(input_dim=X_train.shape[1], num_classes=y_train.shape[1])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
    ]

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    probs = model.predict(X_test, verbose=0)
    pred_indices = np.argmax(probs, axis=1)
    y_true_indices = np.argmax(y_test, axis=1)
    test_accuracy = accuracy_score(y_true_indices, pred_indices)

    model.save(ANN_MODEL_PATH)

    metrics = {
        "test_accuracy": float(test_accuracy),
        "num_classes": float(len(artifacts.label_encoder.classes_)),
        "gpu_used": float(1 if gpu_available else 0),
    }

    print(f"ANN test accuracy: {test_accuracy:.4f}")
    print(f"ANN model saved at: {ANN_MODEL_PATH}")
    print(f"GPU detected by TensorFlow: {gpu_available}")
    return model, metrics


if __name__ == "__main__":
    train_ann_model()
