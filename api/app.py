"""Flask API for Smart Crop Recommendation inference."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from explainability.shap_explainer import CropShapExplainer
from models.ensemble_model import EnsemblePredictor
from models.weather_lstm import forecast_next_weather
from utils.helpers import FEATURE_COLUMNS, prepare_feature_frame

app = Flask(__name__)

_predictor: EnsemblePredictor | None = None
_explainer: CropShapExplainer | None = None


def get_predictor() -> EnsemblePredictor:
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
    return _predictor


def get_explainer() -> CropShapExplainer:
    global _explainer
    if _explainer is None:
        _explainer = CropShapExplainer()
    return _explainer


def _apply_weather_forecast_if_requested(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Optionally override temperature/rainfall via LSTM weather forecast."""
    use_forecast = bool(payload.get("use_weather_forecast", False))
    historical = payload.get("historical_weather", [])

    if not use_forecast:
        return payload

    if not isinstance(historical, list) or len(historical) == 0:
        raise ValueError(
            "historical_weather must be a non-empty list when use_weather_forecast is true"
        )

    history_df = pd.DataFrame(historical)
    if "temperature" not in history_df.columns or "rainfall" not in history_df.columns:
        raise ValueError("Each historical_weather item must include temperature and rainfall")

    forecast_df = forecast_next_weather(history_df[["temperature", "rainfall"]], steps=1)
    payload = payload.copy()
    payload["temperature"] = float(forecast_df.iloc[0]["temperature"])
    payload["rainfall"] = float(forecast_df.iloc[0]["rainfall"])
    payload["forecasted_weather"] = {
        "temperature": payload["temperature"],
        "rainfall": payload["rainfall"],
    }
    return payload


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict() -> Any:
    try:
        payload = request.get_json(force=True)
        payload = _apply_weather_forecast_if_requested(payload)

        frame = prepare_feature_frame(payload)
        predictor = get_predictor()
        probs = predictor.predict_proba(frame)
        class_idx = int(probs[0].argmax())
        result = predictor.predict(frame)
        explanation = get_explainer().explain_top_features(payload, predicted_class_index=class_idx, top_k=3)

        response = {
            "recommended_crop": result.recommended_crop,
            "confidence": round(result.confidence, 4),
            "explanation": explanation,
        }

        if "forecasted_weather" in payload:
            response["forecasted_weather"] = payload["forecasted_weather"]

        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": str(exc), "required_features": FEATURE_COLUMNS}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
