"""LSTM model for temperature and rainfall forecasting."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.helpers import (
    DATASET_PATH,
    WEATHER_MODEL_PATH,
    WEATHER_SCALER_PATH,
    configure_tensorflow_gpu,
    ensure_directories,
    set_global_seed,
)

WEATHER_COLUMNS = ["temperature", "rainfall"]


def create_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 2D weather series into supervised learning sequences."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i])
        y.append(data[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_weather_lstm(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Build a compact LSTM for forecasting temperature and rainfall."""
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(2),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_weather_lstm(
    dataset_path: Path = DATASET_PATH,
    lookback: int = 14,
    epochs: int = 80,
    batch_size: int = 32,
    random_state: int = 42,
) -> Tuple[tf.keras.Model, Dict[str, float]]:
    """Train and persist weather forecasting LSTM and scaler artifacts."""
    set_global_seed(random_state)
    ensure_directories()
    gpu_available = configure_tensorflow_gpu()

    df = pd.read_csv(dataset_path)
    for col in WEATHER_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing weather feature in dataset: {col}")

    weather_df = df[WEATHER_COLUMNS].astype(float).copy()
    weather_df = weather_df.interpolate().ffill().bfill()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(weather_df.values)

    X, y = create_sequences(scaled, lookback=lookback)
    if len(X) < 20:
        raise ValueError("Not enough rows to train weather LSTM. Provide a longer weather history.")

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = build_weather_lstm(input_shape=(lookback, len(WEATHER_COLUMNS)))
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        verbose=1,
    )

    eval_loss, eval_mae = model.evaluate(X_test, y_test, verbose=0)

    model.save(WEATHER_MODEL_PATH)
    joblib.dump(scaler, WEATHER_SCALER_PATH)

    print(f"Weather LSTM saved at: {WEATHER_MODEL_PATH}")
    print(f"Weather scaler saved at: {WEATHER_SCALER_PATH}")
    print(f"GPU detected by TensorFlow: {gpu_available}")

    metrics = {
        "test_loss": float(eval_loss),
        "test_mae": float(eval_mae),
        "gpu_used": float(1 if gpu_available else 0),
    }
    return model, metrics


def forecast_next_weather(
    historical_weather: pd.DataFrame,
    steps: int = 1,
    lookback: int = 14,
) -> pd.DataFrame:
    """Forecast next weather rows from recent temperature/rainfall history."""
    if not WEATHER_MODEL_PATH.exists() or not WEATHER_SCALER_PATH.exists():
        raise FileNotFoundError("Weather artifacts not found. Train weather_lstm.py first.")

    model = tf.keras.models.load_model(WEATHER_MODEL_PATH)
    scaler: MinMaxScaler = joblib.load(WEATHER_SCALER_PATH)

    history = historical_weather[WEATHER_COLUMNS].astype(float).copy()
    if len(history) < lookback:
        # Repeat final row until the required lookback length is met.
        final_row = history.iloc[[-1]]
        repeats = [final_row.copy() for _ in range(lookback - len(history))]
        history = pd.concat([history] + repeats, ignore_index=True)

    scaled_history = scaler.transform(history.values)
    sequence = scaled_history[-lookback:].copy()

    preds_scaled = []
    for _ in range(steps):
        pred = model.predict(sequence[np.newaxis, :, :], verbose=0)[0]
        preds_scaled.append(pred)
        sequence = np.vstack([sequence[1:], pred])

    preds = scaler.inverse_transform(np.array(preds_scaled))
    return pd.DataFrame(preds, columns=WEATHER_COLUMNS)


if __name__ == "__main__":
    train_weather_lstm()
