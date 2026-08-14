"""
explainability.py
SHAP-based explainability for the final selected model, using a model-agnostic
KernelExplainer so it works uniformly across DNN, MLP, TabNet, or an ensemble.
"""

import numpy as np


def build_explainer(predict_proba_fn, background_data, random_state=42):
    """predict_proba_fn: function(X) -> array of shape (n, 2) with class probabilities."""
    import shap
    background = shap.sample(background_data, min(50, len(background_data)), random_state=random_state)
    explainer = shap.KernelExplainer(predict_proba_fn, background)
    return explainer


def _select_positive_class(shap_values):
    """Normalize SHAP output across shap versions to a 2D array (n_samples, n_features)
    of contributions toward class 1 (disease present).

    Newer shap versions return an ndarray of shape (n_samples, n_features, n_classes);
    older versions return a list of per-class arrays."""
    if isinstance(shap_values, list):
        return np.array(shap_values[1])
    arr = np.array(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def compute_shap_values(explainer, X_sample, nsamples=100):
    shap_values = explainer.shap_values(X_sample, nsamples=nsamples)
    return _select_positive_class(shap_values)


def top_contributing_features(explainer, patient_transformed, feature_names, top_n=5, nsamples=100):
    """Return a list of (feature_name, shap_value) for a single patient row."""
    try:
        sv = explainer.shap_values(patient_transformed.reshape(1, -1), nsamples=nsamples)
        sv = _select_positive_class(sv).flatten()
        order = np.argsort(-np.abs(sv))[:top_n]
        return [(feature_names[i], float(sv[i])) for i in order]
    except Exception:
        return []
