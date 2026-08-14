"""
data_utils.py
Loading, cleaning, and stratified sampling utilities for the heart disease dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET_SUBSET_SIZE = 1300
POSSIBLE_TARGET_NAMES = ['target', 'heartdisease', 'num', 'output', 'class', 'label', 'diagnosis']


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load a heart disease CSV from disk."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    """Auto-detect the target/label column; falls back to the last column."""
    for c in df.columns:
        if c.strip().lower() in POSSIBLE_TARGET_NAMES:
            return c
    return df.columns[-1]


def binarize_target_if_needed(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """If the target has more than 2 classes (e.g. UCI 'num' 0-4 severity), binarize to 0/1."""
    df = df.copy()
    if df[target_col].nunique() > 2:
        df[target_col] = (df[target_col] > 0).astype(int)
    return df


def clean_dataset(df: pd.DataFrame) -> dict:
    """Drop exact duplicate rows (important: this dataset is a resampled/expanded
    version of the small UCI Cleveland set and contains many repeated rows —
    keeping duplicates would let the same patient leak across train/test)."""
    original_count = len(df)
    target_col = detect_target_column(df)
    class_dist_before = df[target_col].value_counts().to_dict()
    
    df_clean = df.drop_duplicates().reset_index(drop=True)
    unique_count = len(df_clean)
    duplicate_count = original_count - unique_count
    class_dist_after = df_clean[target_col].value_counts().to_dict()
    
    return {
        "df": df_clean,
        "original_count": original_count,
        "unique_count": unique_count,
        "duplicate_count": duplicate_count,
        "class_dist_before": class_dist_before,
        "class_dist_after": class_dist_after
    }


def stratified_subset(df: pd.DataFrame, target_col: str,
                       target_size: int = TARGET_SUBSET_SIZE,
                       random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return a class-stratified subset of ~target_size rows if df is larger
    than 1500 rows; otherwise return df unchanged."""
    if len(df) <= 1500:
        return df.reset_index(drop=True)
    frac = target_size / len(df)
    df_sub, _ = train_test_split(
        df, train_size=frac, stratify=df[target_col], random_state=random_state
    )
    return df_sub.reset_index(drop=True)


def prepare_dataset(csv_path: str):
    """Full loading pipeline: load -> detect target -> binarize -> dedupe -> subsample.

    Returns (df, target_col, feature_cols, stats_dict).
    """
    df = load_dataset(csv_path)
    target_col = detect_target_column(df)
    df = binarize_target_if_needed(df, target_col)
    
    clean_stats = clean_dataset(df)
    df = clean_stats["df"]
    
    df = stratified_subset(df, target_col)
    feature_cols = [c for c in df.columns if c != target_col]
    
    stats_dict = {
        "original_count": clean_stats["original_count"],
        "unique_count": clean_stats["unique_count"],
        "duplicate_count": clean_stats["duplicate_count"],
        "class_dist_before": clean_stats["class_dist_before"],
        "class_dist_after": clean_stats["class_dist_after"],
        "final_count": len(df)
    }
    
    return df, target_col, feature_cols, stats_dict


def split_dataset(df: pd.DataFrame, target_col: str, feature_cols: list,
                   random_state: int = RANDOM_STATE):
    """60/20/20 stratified train/val/test split."""
    X = df[feature_cols]
    y = df[target_col].astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
