"""
models.py
Model builders: DNN (Keras), MLP (sklearn, tuned), TabNet (optional), and
fast classical baselines (Logistic Regression, Random Forest).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# DNN (Keras)
# ---------------------------------------------------------------------------

def build_dnn(input_dim: int):
    """Compact feed-forward DNN. Kept small deliberately — this dataset has only
    a few hundred training rows, so a very deep/wide network would overfit."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(RANDOM_STATE)
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_dnn(X_train, y_train, X_val, y_val, epochs=150, batch_size=32, patience=15, verbose=0):
    from tensorflow.keras import callbacks

    model = build_dnn(X_train.shape[1])
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )
    return model, history


# ---------------------------------------------------------------------------
# MLP (sklearn, hyperparameter-tuned)
# ---------------------------------------------------------------------------

MLP_PARAM_GRID = {
    'hidden_layer_sizes': [(32,), (64, 32), (64, 64), (128, 64)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.005, 0.01],
}


def train_mlp(X_train, y_train, n_iter=12, cv_splits=5, random_state=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        MLPClassifier(max_iter=1000, random_state=random_state),
        param_distributions=MLP_PARAM_GRID, n_iter=n_iter, cv=skf,
        scoring='accuracy', random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


# ---------------------------------------------------------------------------
# TabNet (optional — only used if pytorch-tabnet installs cleanly)
# ---------------------------------------------------------------------------

def try_import_tabnet():
    try:
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        torch.manual_seed(RANDOM_STATE)
        return TabNetClassifier
    except Exception:
        return None


def train_tabnet(X_train, y_train, X_val, y_val, max_epochs=150, patience=20):
    """Returns (model, val_accuracy) or (None, None) if TabNet is unavailable
    or fails to train in this environment."""
    TabNetClassifier = try_import_tabnet()
    if TabNetClassifier is None:
        return None, None
    try:
        from sklearn.metrics import accuracy_score
        model = TabNetClassifier(
            n_d=16, n_a=16, n_steps=4, gamma=1.5,
            seed=RANDOM_STATE, verbose=0,
            optimizer_params=dict(lr=2e-2)
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['accuracy'],
            max_epochs=max_epochs, patience=patience,
            batch_size=64, virtual_batch_size=16,
            drop_last=False
        )
        val_acc = accuracy_score(y_val, model.predict(X_val))
        return model, val_acc
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Fast classical baselines (context only, not the primary models)
# ---------------------------------------------------------------------------

def train_baselines(X_train, y_train, random_state=RANDOM_STATE):
    baselines = {
        'LogisticRegression': LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=random_state
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced', random_state=random_state
        ),
    }
    for clf in baselines.values():
        clf.fit(X_train, y_train)
    return baselines
