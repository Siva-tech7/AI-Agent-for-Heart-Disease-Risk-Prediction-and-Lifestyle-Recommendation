"""
app.py
Streamlit UI for the "AI Agent for Heart Disease Risk Prediction and
Lifestyle Recommendation" project.

Run with:
    streamlit run app.py

Requires that artifacts/ already contains the outputs of `python train.py`.
"""

import json
import os

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from src.agent import HeartDiseaseAgent, categorize_risk, lifestyle_recommendations
from src.explainability import build_explainer

ARTIFACTS_DIR = "artifacts"

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting ECG Results (0-2)",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise-Induced Angina (1 = yes, 0 = no)",
    "oldpeak": "ST Depression Induced by Exercise",
    "slope": "Slope of Peak Exercise ST Segment (0-2)",
    "ca": "Number of Major Vessels Colored by Fluoroscopy (0-4)",
    "thal": "Thalassemia (0-3)",
}

BINARY_FIELDS = {"sex", "fbs", "exang"}
CATEGORICAL_OPTIONS = {
    "cp": {0: "0 — Typical angina", 1: "1 — Atypical angina", 2: "2 — Non-anginal pain", 3: "3 — Asymptomatic"},
    "restecg": {0: "0 — Normal", 1: "1 — ST-T wave abnormality", 2: "2 — Left ventricular hypertrophy"},
    "slope": {0: "0 — Upsloping", 1: "1 — Flat", 2: "2 — Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"},
    "thal": {0: "0 — Normal (0)", 1: "1 — Normal", 2: "2 — Fixed defect", 3: "3 — Reversable defect"},
}


# ---------------------------------------------------------------------------
# Artifact loading (cached so the app stays fast)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "metadata.json")):
        return None

    with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))

    models = {}
    dnn_path = os.path.join(ARTIFACTS_DIR, "dnn_model.keras")
    if os.path.exists(dnn_path):
        import tensorflow as tf
        models['DNN'] = tf.keras.models.load_model(dnn_path)

    mlp_path = os.path.join(ARTIFACTS_DIR, "mlp_model.joblib")
    if os.path.exists(mlp_path):
        models['MLP'] = joblib.load(mlp_path)

    tabnet_path = os.path.join(ARTIFACTS_DIR, "tabnet_model.zip")
    if os.path.exists(tabnet_path):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            tabnet_model = TabNetClassifier()
            tabnet_model.load_model(tabnet_path)
            models['TabNet'] = tabnet_model
        except Exception:
            pass

    for fname in os.listdir(ARTIFACTS_DIR):
        if fname.startswith("baseline_") and fname.endswith(".joblib"):
            name = fname[len("baseline_"):-len(".joblib")]
            models[name] = joblib.load(os.path.join(ARTIFACTS_DIR, fname))

    background = None
    background_path = os.path.join(ARTIFACTS_DIR, "shap_background.npy")
    if os.path.exists(background_path):
        background = np.load(background_path)

    return {
        "metadata": metadata, "preprocessor": preprocessor,
        "models": models, "background": background,
    }


def make_predict_proba_fn(metadata, models):
    best_model_name = metadata["best_model_name"]
    ensemble_members = metadata.get("ensemble_members", [])

    def single_model_proba(name, X):
        model = models[name]
        if name == 'DNN':
            p1 = model.predict(X, verbose=0).flatten()
        else:
            p1 = model.predict_proba(X)[:, 1]
        return p1

    def predict_proba(X):
        if best_model_name == 'Ensemble':
            p1 = np.mean([single_model_proba(n, X) for n in ensemble_members if n in models], axis=0)
        else:
            p1 = single_model_proba(best_model_name, X)
        return np.column_stack([1 - p1, p1])

    return predict_proba


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Heart Disease AI Agent",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ AI Agent for Heart Disease Risk Prediction and Lifestyle Recommendation")
st.caption("Educational prediction system — **not** a medical diagnostic device. "
           "Always consult a licensed physician for real medical decisions. "
           "These ranges are data-entry safeguards, not clinical recommendations.")

artifacts = load_artifacts()

if artifacts is None:
    st.error(
        "No trained model found. Please run `python train.py --data data/heart.csv` "
        "first to generate the `artifacts/` folder, then restart this app."
    )
    st.stop()

metadata = artifacts["metadata"]
preprocessor = artifacts["preprocessor"]
models = artifacts["models"]
background = artifacts["background"]
feature_cols = metadata["feature_cols"]
feature_names_out = metadata["feature_names_out"]
best_model_name = metadata["best_model_name"]

predict_proba_fn = make_predict_proba_fn(metadata, models)

explainer = None
if background is not None:
    try:
        explainer = build_explainer(predict_proba_fn, background)
    except Exception:
        explainer = None

agent = HeartDiseaseAgent(
    preprocessor=preprocessor,
    predict_proba_fn=predict_proba_fn,
    feature_cols=feature_cols,
    feature_names_out=feature_names_out,
    model_name=best_model_name,
    explainer=explainer,
)

tab_predict, tab_performance, tab_about = st.tabs(
    ["🩺 Patient Risk Prediction", "📊 Model Performance", "ℹ️ About This Project"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Patient prediction (the AI Agent workflow)
# ---------------------------------------------------------------------------

with tab_predict:
    st.subheader("Patient Input")
    st.write("Enter patient details below. Values map to the standard heart-disease "
             "clinical feature set used to train the model.")

    with st.form("patient_form"):
        cols = st.columns(3)
        patient_input = {}

        default_values = {
            "age": 54, "sex": 1, "cp": 0, "trestbps": 130, "chol": 246,
            "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
            "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2,
        }

        for i, feat in enumerate(feature_cols):
            col = cols[i % 3]
            label = FEATURE_LABELS.get(feat, feat)
            default = default_values.get(feat, 0)

            with col:
                if feat in BINARY_FIELDS:
                    patient_input[feat] = st.selectbox(label, options=[0, 1], index=int(default), key=feat)
                elif feat in CATEGORICAL_OPTIONS:
                    options_dict = CATEGORICAL_OPTIONS[feat]
                    selected_label = st.selectbox(
                        label, options=list(options_dict.values()),
                        index=min(int(default), len(options_dict) - 1), key=feat
                    )
                    # reverse mapping from label to int
                    patient_input[feat] = next(k for k, v in options_dict.items() if v == selected_label)
                elif feat == "oldpeak":
                    patient_input[feat] = st.number_input(label, min_value=0.0, max_value=10.0,
                                                            value=float(default), step=0.1, key=feat)
                elif feat == "age":
                    patient_input[feat] = st.number_input(label, min_value=1, max_value=120,
                                                            value=int(default), step=1, key=feat)
                elif feat == "trestbps":
                    patient_input[feat] = st.number_input(label, min_value=50, max_value=250,
                                                            value=int(default), step=1, key=feat)
                elif feat == "chol":
                    patient_input[feat] = st.number_input(label, min_value=50, max_value=700,
                                                            value=int(default), step=1, key=feat)
                elif feat == "thalach":
                    patient_input[feat] = st.number_input(label, min_value=50, max_value=250,
                                                            value=int(default), step=1, key=feat)
                else:
                    patient_input[feat] = st.number_input(label, value=float(default), key=feat)

        explain_toggle = st.checkbox("Include SHAP feature-contribution explanation (slower)", value=True)
        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

    if submitted:
        with st.spinner("Running the AI agent workflow: preprocessing → prediction → explanation..."):
            report = agent.predict(patient_input, explain=explain_toggle and explainer is not None)

        st.divider()
        st.subheader("Risk Report")

        risk_color = {"Low": "green", "Moderate": "orange", "High": "red"}[report["risk_category"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", report["predicted_label"])
        c2.metric("Probability of Disease", f"{report['probability']*100:.1f}%")
        c3.markdown(f"### Risk Category: :{risk_color}[{report['risk_category']}]")

        st.progress(min(max(report["probability"], 0.0), 1.0))

        if report["top_contributing_features"]:
            st.subheader("Top Contributing Features (SHAP)")
            contrib_df = pd.DataFrame(
                report["top_contributing_features"], columns=["Feature", "SHAP Value"]
            )
            st.bar_chart(contrib_df.set_index("Feature"))
            st.caption("Positive values push the prediction toward higher disease risk; "
                       "negative values push toward lower risk.")

        st.subheader("Lifestyle Recommendations")
        for rec in report["lifestyle_recommendations"]:
            st.markdown(f"- {rec}")

        st.info(f"Model used: **{report['model_used']}**  |  {report['disclaimer']}")

        with st.expander("Raw JSON report"):
            st.json(report)

# ---------------------------------------------------------------------------
# TAB 2 — Model performance
# ---------------------------------------------------------------------------

with tab_performance:
    st.subheader("Model Comparison")
    comparison_path = os.path.join(ARTIFACTS_DIR, "comparison_table.csv")
    if os.path.exists(comparison_path):
        comp_df = pd.read_csv(comparison_path)
        st.dataframe(comp_df, use_container_width=True)
        chart_df = comp_df.set_index("Model")[["Validation Accuracy"]]
        st.bar_chart(chart_df)
    else:
        st.warning("comparison_table.csv not found in artifacts/.")

    st.subheader(f"Final Test-Set Metrics — {best_model_name}")
    test_metrics = metadata.get("test_metrics", {})
    if test_metrics:
        mcols = st.columns(len(test_metrics))
        for (name, value), col in zip(test_metrics.items(), mcols):
            col.metric(name.upper(), f"{value:.4f}")

    col_a, col_b = st.columns(2)
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    roc_path = os.path.join(ARTIFACTS_DIR, "roc_curve.png")
    if os.path.exists(cm_path):
        col_a.image(cm_path, caption="Confusion Matrix", use_container_width=True)
    if os.path.exists(roc_path):
        col_b.image(roc_path, caption="ROC Curve", use_container_width=True)

    st.subheader("SHAP Global Explainability")
    col_c, col_d = st.columns(2)
    shap_summary_path = os.path.join(ARTIFACTS_DIR, "shap_summary.png")
    shap_bar_path = os.path.join(ARTIFACTS_DIR, "shap_bar.png")
    if os.path.exists(shap_summary_path):
        col_c.image(shap_summary_path, caption="SHAP Summary Plot", use_container_width=True)
    if os.path.exists(shap_bar_path):
        col_d.image(shap_bar_path, caption="SHAP Feature Importance (bar)", use_container_width=True)
    if not os.path.exists(shap_summary_path):
        st.caption("SHAP plots were not generated during training in this environment.")

# ---------------------------------------------------------------------------
# TAB 3 — About
# ---------------------------------------------------------------------------

with tab_about:
    st.subheader("Project Overview")
    st.markdown(f"""
**Title:** AI Agent for Heart Disease Risk Prediction and Lifestyle Recommendation

**Primary models:** DNN (Keras), TabNet, MLP — plus fast Logistic Regression / Random
Forest baselines used only for context comparison.

**Dataset:** Heart disease clinical dataset (Kaggle: johnsmith88/heart-disease-dataset,
a resampled version of the UCI Cleveland dataset). Deduplicated before training so no
patient record can leak across train/validation/test splits.

**Final selected model:** `{best_model_name}` (selected by validation accuracy,
confirmed once on a held-out test set never used during training or tuning).

**Workflow:** Patient Input → Preprocessing (leak-free `ColumnTransformer`) →
Risk Prediction → Probability → Risk Category → SHAP Explanation → Lifestyle
Recommendation.

**Honesty guarantees:**
- No duplicated test samples, no target leakage, no repeated tuning against the test set.
- Reported accuracy is whatever the held-out test set genuinely produced — never adjusted upward.

⚠️ **This system is for educational purposes only and is not a medical diagnostic
device.** Predictions and recommendations should never replace professional medical
advice, diagnosis, or treatment.
""")

    st.subheader("Feature Glossary")
    glossary_df = pd.DataFrame(
        [(k, v) for k, v in FEATURE_LABELS.items() if k in feature_cols],
        columns=["Column", "Meaning"]
    )
    st.dataframe(glossary_df, use_container_width=True)
