# System Design Document

# 1. System Architecture

The system follows a modular machine learning architecture.

User Input
      |
      v
Data Preprocessing
      |
      v
Weather Forecast Model (LSTM)
      |
      v
Crop Prediction Models
  |           |
  v           v
ANN        Random Forest
      |
      v
Ensemble Voting Layer
      |
      v
Explainable AI Layer (SHAP)
      |
      v
Profit Optimization
      |
      v
Final Recommendation Dashboard

---

# 2. Data Pipeline

## Data Collection

Dataset Source

Kaggle Crop Recommendation Dataset

Features

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

Target

- Crop label (22 crops)

---

## Data Preprocessing

Steps:

1. Data cleaning
2. Missing value handling
3. Feature scaling
4. Normalization
5. Train-test split

---

# 3. Machine Learning Models

## Artificial Neural Network (ANN)

Input Layer

7 neurons

Hidden Layers

ReLU activation

Output Layer

22 neurons

Activation

Softmax

Loss Function

Categorical Cross Entropy

Optimizer

Adam

---

## Random Forest Model

Used to improve prediction robustness.

Advantages

- Handles non-linear relationships
- Resistant to overfitting
- Provides feature importance

---

# 4. Ensemble Layer

Combines predictions from:

- ANN
- Random Forest

Technique

Weighted probability fusion.

Final prediction is selected based on highest confidence.

---

# 5. Explainable AI Layer

Uses SHAP values.

Purpose:

- Show which features influenced prediction.
- Improve model transparency.

Example output:

High rainfall → +0.32
Optimal temperature → +0.21
High humidity → +0.18

---

# 6. Profit Optimization Model

Inputs

- Crop price data
- Expected yield
- Market demand

Output

Profit score for each crop.

Final system recommends:

- Soil best crop
- Profit best crop
- Balanced recommendation

---

# 7. Deployment Architecture

Frontend

Web dashboard

Backend

Python ML API

Model Serving

Flask or FastAPI

Prediction API endpoint used by frontend interface.