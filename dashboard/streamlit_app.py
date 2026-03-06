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
            --bg-1: #000000;
            --bg-2: #0a0a0a;
            --ink: #ffffff;
            --muted: #d0d0d0;
            --brand: #00d1b2;
            --brand-2: #f59e0b;
            --card: #101010;
        }

        .stApp {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
            background: radial-gradient(circle at 15% 20%, rgba(0, 209, 178, 0.12) 0%, transparent 30%),
                        radial-gradient(circle at 88% 12%, rgba(245, 158, 11, 0.10) 0%, transparent 28%),
                        linear-gradient(170deg, var(--bg-1) 0%, var(--bg-2) 100%);
        }

        .stApp, .stApp p, .stApp li, .stApp span, .stApp div,
        .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stMarkdown, .stText, .stCaption {
            color: var(--ink);
        }

        section[data-testid="stSidebar"] {
            background: #050505;
            border-right: 1px solid rgba(255, 255, 255, 0.10);
        }

        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            background: #0b0b0b;
            box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.06), 0 12px 30px rgba(0, 0, 0, 0.45);
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
            border: 1px solid rgba(255, 255, 255, 0.20);
            background: var(--card);
            padding: 0.9rem 1rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
        }

        .metric-card h3 {
            color: #ffffff;
            margin-top: 0.25rem;
            margin-bottom: 0.1rem;
            font-size: 1.4rem;
        }

        .mono {
            font-family: 'IBM Plex Mono', monospace;
            color: #d4d4d4;
            letter-spacing: 0.3px;
        }

        button[kind="primary"] {
            border: 1px solid rgba(255, 255, 255, 0.25) !important;
            background: linear-gradient(135deg, #00d1b2, #0ea5e9) !important;
            color: #000000 !important;
            font-weight: 700 !important;
        }

        .stTextInput input, .stTextArea textarea {
            background-color: #0d0d0d !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.20) !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background: #0d0d0d !important;
            color: #ffffff !important;
        }

        [data-testid="stDataFrame"] {
            background-color: #0d0d0d !important;
            color: #ffffff !important;
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
    colors = ["#00d1b2" if val >= 0 else "#f43f5e" for val in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#070707")
    ax.set_facecolor("#111111")

    ax.barh(features, values, color=colors)
    ax.set_title("Top Feature Contributions", color="#ffffff", pad=12)
    ax.set_xlabel("Contribution", color="#eaeaea")
    ax.tick_params(axis="x", colors="#dcdcdc")
    ax.tick_params(axis="y", colors="#ffffff")
    ax.grid(axis="x", color="#2b2b2b", linestyle="--", linewidth=0.8, alpha=0.8)

    for spine in ax.spines.values():
        spine.set_color("#2a2a2a")

    for index, value in enumerate(values):
        ax.text(
            value + (0.01 if value >= 0 else -0.01),
            index,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            color="#f5f5f5",
            fontsize=9,
        )

    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_probability_distribution(top_probs: Dict[str, float]) -> plt.Figure:
    """Create a polished dark chart for class probabilities."""
    labels = list(top_probs.keys())
    probs = [float(top_probs[name]) for name in labels]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.patch.set_facecolor("#070707")
    ax.set_facecolor("#111111")

    palette = ["#38bdf8", "#00d1b2", "#22c55e", "#f59e0b", "#f97316", "#f43f5e", "#a78bfa", "#eab308"]
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    bars = ax.bar(labels, probs, color=colors, edgecolor="#1f1f1f", linewidth=1.0)
    ax.set_title("Top Crop Probability Scores", color="#ffffff", pad=12)
    ax.set_ylabel("Probability", color="#eaeaea")
    ax.tick_params(axis="x", colors="#ffffff", rotation=20)
    ax.tick_params(axis="y", colors="#dcdcdc")
    ax.set_ylim(0, max(probs) * 1.22 if probs else 1)
    ax.grid(axis="y", color="#2b2b2b", linestyle="--", linewidth=0.8, alpha=0.8)

    for spine in ax.spines.values():
        spine.set_color("#2a2a2a")

    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{prob:.2%}",
            ha="center",
            va="bottom",
            color="#f8f8f8",
            fontsize=9,
        )

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
            styled_df = prob_df.copy()
            styled_df["probability"] = styled_df["probability"].map(lambda val: f"{val:.2%}")
            st.table(styled_df)
            st.pyplot(plot_probability_distribution(top_probs))
        else:
            st.info("Probability distribution is available only in Local Model mode.")

    with raw_tab:
        st.json(result)


st.set_page_config(page_title="Smart Crop Recommendation", page_icon="CR", layout="wide")
inject_custom_css()
render_title()

with st.sidebar:
    st.header("Inference Settings")
    st.caption("Recommended for production: Flask API mode")
    mode = st.radio("Run with", ["Flask API", "Local Models"], index=0)
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
