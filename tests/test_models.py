import numpy as np
import pandas as pd
from src.models import build_dnn, train_mlp, train_baselines
from src.preprocessing import build_preprocessor, to_dense

def test_dnn_shape():
    model = build_dnn(input_dim=5)
    X = np.random.rand(10, 5)
    y_pred = model.predict(X, verbose=0)
    assert y_pred.shape == (10, 1)

def test_baselines_proba_shape():
    X_train = np.random.rand(20, 5)
    y_train = np.random.randint(0, 2, 20)
    baselines = train_baselines(X_train, y_train)
    rf = baselines['RandomForest']
    proba = rf.predict_proba(X_train)
    assert proba.shape == (20, 2)
