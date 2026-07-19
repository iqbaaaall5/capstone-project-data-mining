"""
train_model.py
Script training dan hyperparameter tuning untuk 3 model klasifikasi:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost (model terbaik)
Tuning menggunakan RandomizedSearchCV.
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import preprocess_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5


def train_logistic_regression(X_train, y_train):
    print("\n[1/3] Training Logistic Regression...")
    param_dist = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [500, 1000],
    }
    base = LogisticRegression(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=10, cv=cv,
        scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def train_random_forest(X_train, y_train):
    print("\n[2/3] Training Random Forest...")
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=15, cv=cv,
        scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def train_xgboost(X_train, y_train):
    print("\n[3/3] Training XGBoost...")
    scale_pos = int(np.sum(y_train == 0) / np.sum(y_train == 1))
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
    }
    base = XGBClassifier(
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        verbosity=0,
        use_label_encoder=False,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=20, cv=cv,
        scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def run_training():
    # Preprocessing
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_names, scaler, encoders
    ) = preprocess_pipeline()

    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Simpan semua model
    models = {
        "logistic_regression": lr_model,
        "random_forest": rf_model,
        "xgboost": xgb_model,
    }
    for name, model in models.items():
        joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")
        print(f"  Saved: {MODEL_DIR}/{name}.pkl")

    # Simpan best model (XGBoost) sebagai best_model.pkl
    joblib.dump(xgb_model, f"{MODEL_DIR}/best_model.pkl")
    print(f"\nBest model (XGBoost) disimpan sebagai: {MODEL_DIR}/best_model.pkl")

    print("\n=== Training Selesai ===")
    return models, X_train, X_val, X_test, y_train, y_val, y_test, feature_names


if __name__ == "__main__":
    run_training()
