import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "model/model.pkl")
scaler = joblib.load(BASE_DIR / "model/scaler.pkl")

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

def predict_disease(data: dict) -> dict:
    features = np.array([[data[f] for f in FEATURE_NAMES]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]

    return {
        "prediction": int(prediction),
        "result": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
        "confidence": f"{max(probability) * 100:.2f}%",
        "probabilities": {
            "no_disease": f"{probability[0] * 100:.2f}%",
            "disease": f"{probability[1] * 100:.2f}%"
        }
    }