"""Production-style Streamlit frontend for crop recommendation."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from explainability.shap_explainer import CropShapExplainer
from models.ensemble_model import EnsemblePredictor
from models.weather_lstm import forecast_next_weather

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def inject_custom_css() -> None:
    """Apply a custom visual system for a polished dashboard look."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg-1: #f4f7ee;
            --bg-2: #e5edcf;
            --ink: #1f2c1d;
            --muted: #4f5f4b;
            --brand: #2e6f40;
            --brand-2: #c8a24a;
            --card: #ffffff;
        }

        .stApp {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
            background: radial-gradient(circle at top right, #dce8b8 0%, transparent 38%),
                        radial-gradient(circle at 20% 20%, #f3e5ba 0%, transparent 32%),
                        linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 100%);
        }

        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            border: 1px solid rgba(46, 111, 64, 0.2);
            background: linear-gradient(140deg, rgba(46, 111, 64, 0.08), rgba(200, 162, 74, 0.1));
            animation: riseIn 450ms ease-out;
        }

        .hero h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.8rem;
            margin: 0 0 0.3rem 0;
        }

        .hero p {
            margin: 0;
            color: var(--muted);
        }

        .metric-card {
            border-radius: 14px;
            border: 1px solid rgba(46, 111, 64, 0.18);
            background: var(--card);
            padding: 0.9rem 1rem;
        }

        .mono {
            font-family: 'IBM Plex Mono', monospace;
        }

        @keyframes riseIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 900px) {
            .hero h1 { font-size: 1.35rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_predictor() -> EnsemblePredictor:
    """Load local ensemble predictor once per session."""
    return EnsemblePredictor()


@st.cache_resource
def load_explainer() -> CropShapExplainer:
    """Load SHAP explainer once per session."""
    return CropShapExplainer()


def parse_weather_csv(csv_text: str) -> pd.DataFrame:
    """Parse and validate weather history CSV."""
    history_df = pd.read_csv(StringIO(csv_text.strip()))
    required = ["temperature", "rainfall"]
    missing = [col for col in required if col not in history_df.columns]
    if missing:
        raise ValueError(f"Historical weather CSV missing columns: {missing}")
    return history_df[required].astype(float)


def post_to_api(api_url: str, payload: Dict[str, object]) -> Dict[str, object]:
    """Call Flask prediction API and return parsed JSON response."""
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to API at {api_url}: {exc.reason}") from exc


def plot_explanation(explanation: Dict[str, float]) -> plt.Figure:
    """Create contribution chart from feature importance dictionary."""
    features = list(explanation.keys())
    values = [float(explanation[name]) for name in features]
    colors = ["#2e6f40" if val >= 0 else "#9c2f2f" for val in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(features, values, color=colors)
    ax.set_title("Top Feature Contributions")
    ax.set_xlabel("Contribution")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def local_predict(
    payload: Dict[str, float],
    use_weather_forecast: bool,
    weather_csv: str,
    top_k: int,
) -> Tuple[Dict[str, object], Optional[pd.DataFrame]]:
    """Run local predictor path and return API-shaped response."""
    local_payload = payload.copy()
    forecast_df: Optional[pd.DataFrame] = None

    if use_weather_forecast:
        history_df = parse_weather_csv(weather_csv)
        forecast_df = forecast_next_weather(history_df, steps=1)
        local_payload["temperature"] = float(forecast_df.iloc[0]["temperature"])
        local_payload["rainfall"] = float(forecast_df.iloc[0]["rainfall"])

    predictor = load_predictor()
    result = predictor.predict(local_payload)

    class_order = list(result.probabilities.keys())
    class_idx = class_order.index(result.recommended_crop)
    explanation = load_explainer().explain_top_features(
        local_payload,
        predicted_class_index=class_idx,
        top_k=top_k,
    )

    response: Dict[str, object] = {
        "recommended_crop": result.recommended_crop,
        "confidence": result.confidence,
        "explanation": explanation,
        "top_probabilities": dict(
            sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ),
    }
    return response, forecast_df


def render_title() -> None:
    """Render branded header section."""
    st.markdown(
        """
        <div class="hero">
          <h1>Smart Crop Recommendation and Decision Support</h1>
          <p>Hybrid ANN + Random Forest ensemble with weather-forecast assist and SHAP explainability.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(result: Dict[str, object], forecast_df: Optional[pd.DataFrame]) -> None:
    """Render prediction output cards and charts."""
    crop = str(result.get("recommended_crop", "unknown"))
    confidence = float(result.get("confidence", 0.0))
    explanation = result.get("explanation", {})
    top_probs = result.get("top_probabilities", {})

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown('<div class="metric-card"><div class="mono">Recommended Crop</div><h3>' + crop + '</h3></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(
            '<div class="metric-card"><div class="mono">Confidence</div><h3>'
            + f"{confidence:.2%}"
            + "</h3></div>",
            unsafe_allow_html=True,
        )

    if forecast_df is not None:
        st.info(
            "Weather forecast applied: "
            f"temperature={forecast_df.iloc[0]['temperature']:.2f} C, "
            f"rainfall={forecast_df.iloc[0]['rainfall']:.2f} mm"
        )

    chart_tab, prob_tab, raw_tab = st.tabs(["Feature Insight", "Class Scores", "Raw Output"])

    with chart_tab:
        if isinstance(explanation, dict) and explanation:
            st.pyplot(plot_explanation({k: float(v) for k, v in explanation.items()}))
        else:
            st.warning("No explanation values were returned.")

    with prob_tab:
        if isinstance(top_probs, dict) and top_probs:
            prob_df = pd.DataFrame(
                {"crop": list(top_probs.keys()), "probability": list(top_probs.values())}
            )
            st.dataframe(prob_df, use_container_width=True)
            st.bar_chart(prob_df.set_index("crop"), y="probability", use_container_width=True)
        else:
            st.info("Probability distribution is available only in Local Model mode.")

    with raw_tab:
        st.json(result)


st.set_page_config(page_title="Smart Crop Recommendation", page_icon="CR", layout="wide")
inject_custom_css()
render_title()

with st.sidebar:
    st.header("Inference Settings")
    mode = st.radio("Run with", ["Local Models", "Flask API"], index=0)
    api_url = st.text_input("API URL", value="http://127.0.0.1:5000/predict", disabled=(mode != "Flask API"))
    top_k = st.slider("Top features/classes to show", min_value=3, max_value=8, value=5)
    use_weather_forecast = st.checkbox("Use weather forecast assist", value=False)

st.subheader("Input Conditions")
col1, col2, col3 = st.columns(3)
with col1:
    n_val = st.slider("Nitrogen (N)", 0, 140, 90)
    p_val = st.slider("Phosphorus (P)", 0, 145, 42)
    k_val = st.slider("Potassium (K)", 0, 205, 43)
with col2:
    temp_val = st.slider("Temperature (C)", 0.0, 50.0, 25.0, 0.1)
    humidity_val = st.slider("Humidity (%)", 0.0, 100.0, 80.0, 0.1)
with col3:
    ph_val = st.slider("Soil pH", 3.0, 10.0, 6.5, 0.1)
    rainfall_val = st.slider("Rainfall (mm)", 0.0, 300.0, 200.0, 0.1)

history_csv = ""
if use_weather_forecast:
    with st.expander("Weather History Input", expanded=True):
        st.caption("Enter historical weather values as CSV with headers: temperature,rainfall")
        history_csv = st.text_area(
            "Historical Weather CSV",
            value="temperature,rainfall\n24.5,180\n25.0,195\n25.3,205\n24.8,198\n25.1,210",
            height=180,
        )

payload: Dict[str, float] = {
    "N": float(n_val),
    "P": float(p_val),
    "K": float(k_val),
    "temperature": float(temp_val),
    "humidity": float(humidity_val),
    "ph": float(ph_val),
    "rainfall": float(rainfall_val),
}

button_col, _ = st.columns([1, 4])
with button_col:
    run_prediction = st.button("Predict", type="primary", use_container_width=True)

if run_prediction:
    try:
        if mode == "Local Models":
            result, forecast = local_predict(
                payload=payload,
                use_weather_forecast=use_weather_forecast,
                weather_csv=history_csv,
                top_k=top_k,
            )
            render_results(result, forecast_df=forecast)
        else:
            api_payload: Dict[str, object] = dict(payload)
            if use_weather_forecast:
                history_df = parse_weather_csv(history_csv)
                api_payload["use_weather_forecast"] = True
                api_payload["historical_weather"] = history_df.to_dict(orient="records")

            api_response = post_to_api(api_url=api_url, payload=api_payload)
            render_results(api_response, forecast_df=None)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
