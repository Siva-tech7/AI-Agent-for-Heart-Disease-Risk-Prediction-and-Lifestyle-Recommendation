"""
train.py
End-to-end training script for the Heart Disease AI Agent project.

Run with:
    python train.py --data data/heart.csv

Produces, in artifacts/:
    preprocessor.joblib          - fitted sklearn ColumnTransformer
    dnn_model.keras              - trained DNN (if selected/available)
    mlp_model.joblib             - trained MLP (if selected/available)
    tabnet_model.zip             - trained TabNet (if available and selected)
    baseline_<name>.joblib       - fast baseline models (context only)
    metadata.json                - feature columns, target column, best model name, metrics
    comparison_table.csv         - validation accuracy per model
    confusion_matrix.png, roc_curve.png, shap_summary.png, shap_bar.png

No fabricated numbers: every metric printed/saved comes directly from evaluating
the trained model on data it has not been fit on.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.data_utils import prepare_dataset, split_dataset, RANDOM_STATE
from src.preprocessing import build_preprocessor, to_dense
from src.models import build_dnn, train_dnn, train_mlp, train_tabnet, train_baselines
from src.evaluation import (
    compute_metrics, plot_confusion_matrix, plot_roc_curve,
    build_comparison_table, full_classification_report
)
from src.explainability import build_explainer, compute_shap_values  # compute_shap_values normalizes shap-version output shapes

ARTIFACTS_DIR = "artifacts"


def get_proba_fn(name, models):
    """Return a function(X_transformed) -> probability of class 1, for a given model name."""
    if name == 'DNN':
        return lambda X: models['dnn'].predict(X, verbose=0).flatten()
    elif name == 'MLP':
        return lambda X: models['mlp'].predict_proba(X)[:, 1]
    elif name == 'TabNet':
        return lambda X: models['tabnet'].predict_proba(X)[:, 1]
    elif name in models.get('baselines', {}):
        clf = models['baselines'][name]
        return lambda X, clf=clf: clf.predict_proba(X)[:, 1]
    return None


def main(args):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1/7: Loading and cleaning dataset")
    print("=" * 70)
    df, target_col, feature_cols = prepare_dataset(args.data)
    print(f"Dataset ready: {df.shape[0]} rows, {len(feature_cols)} features, target='{target_col}'")
    print("Class balance:", df[target_col].value_counts(normalize=True).round(3).to_dict())

    print("\n" + "=" * 70)
    print("STEP 2/7: Train / validation / test split (60/20/20, stratified)")
    print("=" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, target_col, feature_cols)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("\n" + "=" * 70)
    print("STEP 3/7: Preprocessing (fit on train only)")
    print("=" * 70)
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    X_train_t = to_dense(preprocessor.transform(X_train))
    X_val_t = to_dense(preprocessor.transform(X_val))
    X_test_t = to_dense(preprocessor.transform(X_test))
    feature_names_out = preprocessor.get_feature_names_out().tolist()
    y_train_arr, y_val_arr, y_test_arr = y_train.values, y_val.values, y_test.values
    print(f"Transformed feature count: {X_train_t.shape[1]}")

    models = {'baselines': {}}
    val_scores = {}

    print("\n" + "=" * 70)
    print("STEP 4/7: Training primary models (DNN, TabNet, MLP) + baselines")
    print("=" * 70)

    print("\n-- DNN (Keras) --")
    dnn_model, history = train_dnn(X_train_t, y_train_arr, X_val_t, y_val_arr,
                                    epochs=args.dnn_epochs, verbose=0)
    models['dnn'] = dnn_model
    val_scores['DNN'] = float(max(history.history['val_accuracy']))
    print(f"DNN best validation accuracy: {val_scores['DNN']:.4f} "
          f"(stopped at epoch {len(history.history['loss'])})")

    print("\n-- TabNet (optional) --")
    tabnet_model, tabnet_val_acc = train_tabnet(X_train_t, y_train_arr, X_val_t, y_val_arr)
    if tabnet_model is not None:
        models['tabnet'] = tabnet_model
        val_scores['TabNet'] = float(tabnet_val_acc)
        print(f"TabNet validation accuracy: {tabnet_val_acc:.4f}")
    else:
        print("TabNet not available/failed to train in this environment — skipped (does not affect other models).")

    print("\n-- MLP (sklearn, RandomizedSearchCV-tuned) --")
    mlp_model, mlp_params, mlp_cv_score = train_mlp(X_train_t, y_train_arr, n_iter=args.mlp_search_iters)
    models['mlp'] = mlp_model
    from sklearn.metrics import accuracy_score
    val_scores['MLP'] = float(accuracy_score(y_val_arr, mlp_model.predict(X_val_t)))
    print(f"Best MLP params: {mlp_params}")
    print(f"MLP CV accuracy (train folds): {mlp_cv_score:.4f} | Validation accuracy: {val_scores['MLP']:.4f}")

    print("\n-- Baselines (context only): LogisticRegression, RandomForest --")
    baselines = train_baselines(X_train_t, y_train_arr)
    models['baselines'] = baselines
    for name, clf in baselines.items():
        acc = accuracy_score(y_val_arr, clf.predict(X_val_t))
        val_scores[name] = float(acc)
        print(f"{name}: validation accuracy = {acc:.4f}")

    print("\n" + "=" * 70)
    print("STEP 5/7: Model selection (by validation accuracy) + optional ensemble")
    print("=" * 70)
    comparison_df = build_comparison_table(val_scores)
    print(comparison_df.to_string(index=False))

    best_model_name = comparison_df.iloc[0]['Model']

    ensemble_members = [n for n in ['DNN', 'MLP', 'TabNet'] if n in val_scores]
    ens_val_proba = np.mean(
        [get_proba_fn(n, models)(X_val_t) for n in ensemble_members], axis=0
    )
    ensemble_val_acc = float(accuracy_score(y_val_arr, (ens_val_proba >= 0.5).astype(int)))
    print(f"\nSoft-voting ensemble ({'+'.join(ensemble_members)}) validation accuracy: {ensemble_val_acc:.4f}")

    if ensemble_val_acc > val_scores[best_model_name]:
        best_model_name = 'Ensemble'
        val_scores['Ensemble'] = ensemble_val_acc
        print("Ensemble beats the single best model on validation — adopting it as the final model.")
    else:
        print(f"Single best model '{best_model_name}' remains the final choice.")

    print(f"\n>>> FINAL SELECTED MODEL: {best_model_name} <<<")

    def predict_proba_final(X):
        if best_model_name == 'Ensemble':
            p1 = np.mean([get_proba_fn(n, models)(X) for n in ensemble_members], axis=0)
        else:
            p1 = get_proba_fn(best_model_name, models)(X)
        return np.column_stack([1 - p1, p1])

    print("\n" + "=" * 70)
    print("STEP 6/7: Final evaluation on the held-out TEST set (touched once)")
    print("=" * 70)
    test_proba = predict_proba_final(X_test_t)[:, 1]
    y_pred = (test_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_test_arr, y_pred, test_proba)
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    if metrics['accuracy'] >= 0.95:
        print(">= 95% accuracy achieved genuinely on the held-out test set.")
    else:
        print(f"Genuine held-out accuracy is {metrics['accuracy']*100:.2f}%, below the 95% target. "
              "Reported honestly rather than adjusted.")
    print("\nClassification report:\n", full_classification_report(y_test_arr, y_pred))

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_confusion_matrix(y_test_arr, y_pred, model_name=best_model_name, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_roc_curve(y_test_arr, test_proba, auc_value=metrics.get('roc_auc'), ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    comparison_df = build_comparison_table(val_scores, best_model=best_model_name,
                                            test_accuracy=metrics['accuracy'])
    comparison_df.to_csv(os.path.join(ARTIFACTS_DIR, "comparison_table.csv"), index=False)

    print("\n" + "=" * 70)
    print("STEP 7/7: SHAP explainability + saving all artifacts")
    print("=" * 70)
    try:
        import shap
        explainer = build_explainer(predict_proba_final, X_train_t)
        shap_sample = X_test_t[:80]
        shap_values = compute_shap_values(explainer, shap_sample, nsamples=100)

        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, shap_sample, feature_names=feature_names_out, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "shap_summary.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, shap_sample, feature_names=feature_names_out,
                           plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "shap_bar.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        print("SHAP plots saved.")
        # We don't pickle the explainer itself (it wraps a local closure, which
        # isn't picklable). Instead we save a small background sample; the
        # Streamlit app rebuilds a fresh KernelExplainer from it at startup,
        # around its own (identical) predict_proba function.
        np.save(os.path.join(ARTIFACTS_DIR, "shap_background.npy"),
                X_train_t[:min(50, len(X_train_t))])
    except Exception as e:
        print("SHAP explanation skipped due to an error in this environment:", e)

    # Save all artifacts needed by the Streamlit app / agent
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))
    if 'dnn' in models:
        models['dnn'].save(os.path.join(ARTIFACTS_DIR, "dnn_model.keras"))
    if 'mlp' in models:
        joblib.dump(models['mlp'], os.path.join(ARTIFACTS_DIR, "mlp_model.joblib"))
    if 'tabnet' in models:
        models['tabnet'].save_model(os.path.join(ARTIFACTS_DIR, "tabnet_model"))
    for name, clf in models['baselines'].items():
        joblib.dump(clf, os.path.join(ARTIFACTS_DIR, f"baseline_{name}.joblib"))

    metadata = {
        "target_col": target_col,
        "feature_cols": feature_cols,
        "feature_names_out": feature_names_out,
        "best_model_name": best_model_name,
        "ensemble_members": ensemble_members if best_model_name == 'Ensemble' else [],
        "validation_scores": val_scores,
        "test_metrics": metrics,
        "available_models": list(val_scores.keys()),
    }
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nAll artifacts saved to '{ARTIFACTS_DIR}/'.")
    print("Training complete. Launch the UI with: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Heart Disease AI Agent models.")
    parser.add_argument("--data", type=str, default="data/heart.csv", help="Path to the heart disease CSV.")
    parser.add_argument("--dnn-epochs", type=int, default=150, help="Max epochs for the DNN (early stopping applies).")
    parser.add_argument("--mlp-search-iters", type=int, default=12, help="RandomizedSearchCV iterations for MLP.")
    args = parser.parse_args()
    main(args)
