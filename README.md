# AI Agent for Heart Disease Risk Prediction and Lifestyle Recommendation

An end-to-end, reproducible, and explainable AI workflow for predicting heart disease risk based on clinical features.

**⚠️ DISCLAIMER: This system is for educational and research purposes only. It is not a medical diagnostic device. Predictions and recommendations should never replace professional medical advice, diagnosis, or treatment.**

## Objective
To demonstrate a scientifically honest machine-learning pipeline for medical-risk classification. This project prioritizes:
1. **No Data Leakage**: Exact duplicates are removed before splitting so no patient can leak from train to test.
2. **Honest Evaluation**: All reported metrics come from an untouched test set, without manipulating results to artificially inflate accuracy. 
3. **Consistent Model Interface**: All predictive models expose a unified `(n_samples, 2)` probability array format.
4. **Explainability**: SHAP (SHapley Additive exPlanations) highlights feature contributions transparently.

## Dataset & Data Cleaning
The dataset uses clinical features commonly associated with cardiovascular risk (e.g., Age, Sex, Chest Pain Type, Resting BP, Cholesterol, etc.).

* **Deduplication**: The provided dataset contained numerous duplicate rows. These are removed *before* splitting to ensure patient independence across Train, Validation, and Test sets.

## Architecture
- **Preprocessing**: `ColumnTransformer` handles imputation, scaling (continuous features), and one-hot encoding (categorical features), fitted solely on training data.
- **Models**: Evaluates multiple architectures including DNN, MLP, TabNet (optional), Logistic Regression, and Random Forest.
- **AI Agent Workflow**: 
  1. Validates input
  2. Preprocesses
  3. Predicts Probability
  4. Categorizes Risk (Educational only)
  5. Computes SHAP Explanations
  6. Provides General Wellness Guidance

## Installation & Running

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train models and generate artifacts**:
   ```bash
   python train.py
   ```
4. **Run tests**:
   ```bash
   pytest tests/
   ```
5. **Launch the UI**:
   ```bash
   streamlit run app.py
   ```

## Limitations
- This model is trained on a very small sample size and cannot generalize to real-world populations.
- The risk categories ("Low", "Moderate", "High") are arbitrary educational thresholds and have no clinical validation.
- The explanations provided by SHAP show mathematical model influence, not biological causality.

## Methodology & Results
The test accuracy is completely genuine and reported directly from evaluation on a held-out test split (20% of unique rows). Check `artifacts/metadata.json` or the Model Performance tab in the UI for the true metrics!
