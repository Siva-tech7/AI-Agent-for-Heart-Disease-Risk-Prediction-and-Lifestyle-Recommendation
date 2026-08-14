"""
preprocessing.py
Leak-free preprocessing pipeline built with sklearn ColumnTransformer.
Fit ONLY on training data; reused (never refit) on validation/test/inference data.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build (but do not fit) a ColumnTransformer for numeric, binary, and categorical features."""
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    binary_features = ['sex', 'fbs', 'exang']
    categorical_features = ['cp', 'restecg', 'slope', 'ca', 'thal']

    # Ensure we only use columns that actually exist in X
    cols = X.columns.tolist()
    continuous_features = [c for c in continuous_features if c in cols]
    binary_features = [c for c in binary_features if c in cols]
    categorical_features = [c for c in categorical_features if c in cols]
    
    # Any other columns not classified (fallback)
    classified = set(continuous_features + binary_features + categorical_features)
    other_numeric = [c for c in X.select_dtypes(include=[np.number]).columns if c not in classified]
    other_cat = [c for c in X.select_dtypes(exclude=[np.number]).columns if c not in classified]

    continuous_features.extend(other_numeric)
    categorical_features.extend(other_cat)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, continuous_features),
        ('bin', binary_transformer, binary_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor


def to_dense(arr):
    """Convert a sparse matrix output to a dense numpy array if needed."""
    if hasattr(arr, 'toarray'):
        return arr.toarray()
    return arr
