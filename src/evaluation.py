"""
evaluation.py
Metrics, confusion matrix, ROC curve, and comparison table helpers.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless-safe for scripts; Streamlit overrides as needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    brier_score_loss
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': specificity,
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_proba)
        except Exception:
            metrics['roc_auc'] = np.nan
            metrics['brier_score'] = np.nan
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name='Model', ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_title(f'Confusion Matrix — {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return ax


def plot_roc_curve(y_true, y_proba, auc_value=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    label = f'AUC = {auc_value:.3f}' if auc_value is not None else 'ROC'
    ax.plot(fpr, tpr, label=label, color='#DD8452')
    ax.plot([0, 1], [0, 1], '--', color='grey')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    return ax


def build_comparison_table(val_scores: dict, best_model: str = None, test_accuracy: float = None) -> pd.DataFrame:
    df = pd.DataFrame(
        sorted(val_scores.items(), key=lambda x: -x[1]),
        columns=['Model', 'Validation Accuracy']
    )
    if best_model is not None and test_accuracy is not None:
        df.loc[df['Model'] == best_model, 'Test Accuracy'] = test_accuracy
    return df


def full_classification_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred)
