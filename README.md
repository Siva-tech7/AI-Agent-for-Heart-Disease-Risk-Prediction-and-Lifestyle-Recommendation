# AI Agent for Heart Disease Risk Prediction and Lifestyle Recommendation

An academic mini project: a lightweight, honest, and explainable machine
learning pipeline for heart disease risk prediction, wrapped in an
**AI Agent** workflow and a **Streamlit** UI.

> ⚠️ **Disclaimer:** This is an educational prediction system, **not** a
> medical diagnostic device. Nothing here should be used for real clinical
> decisions.

## Workflow

```
Patient Input → Preprocessing → Risk Prediction → Probability →
Risk Category → SHAP Explanation → Lifestyle Recommendation
```

## Project Structure

```
.
├── app.py                 # Streamlit UI
├── train.py                # End-to-end training script (run this first)
├── requirements.txt
├── data/
│   └── heart.csv           # Heart disease dataset (Kaggle: johnsmith88/heart-disease-dataset)
├── src/
│   ├── data_utils.py        # Loading, cleaning, stratified subsampling, splitting
│   ├── preprocessing.py     # Leak-free ColumnTransformer pipeline
│   ├── models.py            # DNN (Keras), MLP (sklearn), TabNet (optional), baselines
│   ├── evaluation.py        # Metrics, confusion matrix, ROC curve, comparison table
│   ├── explainability.py    # SHAP KernelExplainer helpers
│   └── agent.py              # The AI Agent class (predict -> explain -> recommend)
└── artifacts/                # Created by train.py: trained models, plots, metadata.json
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

TabNet (`torch` + `pytorch-tabnet`) is optional — if it fails to install on
your machine, everything still works: `train.py` automatically detects this
and trains DNN + MLP + baselines only, and reports honestly that TabNet was
skipped.

## 1. Train the models

```bash
python train.py --data data/heart.csv
```

This will:
1. Load and clean `data/heart.csv` (deduplicate — important, since this
   dataset is a resampled version of the small UCI Cleveland set and
   contains many repeated rows).
2. Use a class-stratified subset of ~1000–1500 rows if the source is larger
   than 1500 rows (this dataset, after deduplication, is already small
   enough that no further subsampling is needed).
3. Split 60/20/20 into train / validation / test (stratified).
4. Fit a leak-free preprocessing pipeline (median/mode imputation, scaling,
   one-hot encoding) on the training data only.
5. Train the primary models: **DNN** (Keras, with early stopping),
   **TabNet** (if available), and **MLP** (sklearn, tuned with
   `RandomizedSearchCV`) — plus fast Logistic Regression / Random Forest
   baselines for context.
6. Select the best model by validation accuracy (optionally a soft-voting
   ensemble of the neural models, if it genuinely wins).
7. Evaluate once, honestly, on the untouched test set.
8. Generate SHAP explainability plots.
9. Save everything needed by the UI into `artifacts/`.

Optional flags:

```bash
python train.py --data data/heart.csv --dnn-epochs 150 --mlp-search-iters 12
```

## 2. Launch the Streamlit app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

The app has three tabs:
- **Patient Risk Prediction** — enter a patient's clinical values and get a
  full AI-agent report: prediction, probability, risk category, SHAP
  feature contributions, and lifestyle recommendations.
- **Model Performance** — validation/test metrics, confusion matrix, ROC
  curve, and SHAP global explainability plots generated during training.
- **About This Project** — project summary and feature glossary, useful for
  a viva/presentation.

## Honesty Guarantees

- Preprocessing is fit **only** on the training split — no leakage.
- The test set is evaluated **once**, at the end, and never used for tuning.
- No duplicate samples are allowed to cross the train/val/test boundary.
- Reported accuracy is exactly what the held-out test set produces — never
  adjusted or cherry-picked upward. If it's below 95%, the script says so.

## Dataset

[Heart Disease Dataset — johnsmith88, Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
(1025 rows × 14 columns before cleaning; a resampled version of the UCI
Cleveland heart disease dataset). Target column: `target` (0 = no disease,
1 = disease present).

## Limitations & Future Scope

See the "About This Project" tab in the app, or the conclusion section
printed by `train.py`, for a full discussion suitable for a viva —
including why DNN/TabNet may not clearly beat simpler models on a dataset
this small, and ideas for extending the project (larger datasets,
probability calibration, fairness checks, deployment).
