"""
agent.py
The AI Agent workflow:
Patient Input -> Preprocessing -> Risk Prediction -> Probability ->
Risk Category -> Explanation -> Lifestyle Recommendation

This wraps a trained model + preprocessor + (optional) SHAP explainer into a
single callable that produces a structured, patient-facing report.

IMPORTANT: This is an educational prediction system, NOT a medical diagnostic
device. Nothing it outputs should be used for real clinical decisions.
"""

import numpy as np
import pandas as pd

from .explainability import top_contributing_features


def categorize_risk(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    elif probability < 0.66:
        return "Moderate"
    else:
        return "High"


def lifestyle_recommendations(risk_category: str) -> list:
    base = [
        "This is general wellness information, not medical advice — consult a licensed physician.",
        "Maintain a balanced diet low in saturated fat and added sugar.",
        "Aim for regular moderate exercise as approved by a doctor (e.g. brisk walking).",
        "Avoid tobacco use and limit alcohol intake.",
        "Monitor blood pressure and cholesterol regularly.",
    ]
    if risk_category == "High":
        base.append("Schedule a consultation with a cardiologist promptly for a full clinical evaluation.")
    elif risk_category == "Moderate":
        base.append("Consider a routine check-up with a physician to review cardiovascular risk factors.")
    else:
        base.append("Continue routine annual health check-ups to maintain low risk.")
    return base


class HeartDiseaseAgent:
    """Callable AI agent wrapping preprocessing + model + (optional) explainer."""

    def __init__(self, preprocessor, predict_proba_fn, feature_cols, feature_names_out,
                 model_name: str, explainer=None):
        self.preprocessor = preprocessor
        self.predict_proba_fn = predict_proba_fn  # function(X_transformed) -> (n,2) probs
        self.feature_cols = feature_cols
        self.feature_names_out = feature_names_out
        self.model_name = model_name
        self.explainer = explainer

    def predict(self, patient_dict: dict, explain: bool = True) -> dict:
        patient_df = pd.DataFrame([patient_dict])[self.feature_cols]
        patient_t = self.preprocessor.transform(patient_df)
        if hasattr(patient_t, 'toarray'):
            patient_t = patient_t.toarray()

        proba = self.predict_proba_fn(patient_t)
        probability = float(np.asarray(proba)[0, 1])
        prediction = int(probability >= 0.5)
        risk_category = categorize_risk(probability)

        contributing = []
        if explain and self.explainer is not None:
            contributing = top_contributing_features(
                self.explainer, patient_t[0], self.feature_names_out
            )

        report = {
            "predicted_class": prediction,
            "predicted_label": "Heart Disease Risk Present" if prediction == 1 else "No Heart Disease Risk Detected",
            "probability": round(probability, 4),
            "risk_category": risk_category,
            "top_contributing_features": contributing,
            "lifestyle_recommendations": lifestyle_recommendations(risk_category),
            "model_used": self.model_name,
            "disclaimer": "Educational prediction only — not a medical diagnosis. Consult a licensed physician."
        }
        return report

    def __call__(self, patient_dict: dict, explain: bool = True) -> dict:
        return self.predict(patient_dict, explain=explain)
