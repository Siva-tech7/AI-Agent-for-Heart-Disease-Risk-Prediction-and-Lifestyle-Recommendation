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
    """Build (but do not fit) a ColumnTransformer for numeric + categorical features."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor


def to_dense(arr):
    """Convert a sparse matrix output to a dense numpy array if needed."""
    if hasattr(arr, 'toarray'):
        return arr.toarray()
    return arr
