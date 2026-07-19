"""
evaluate_model.py
Script evaluasi komprehensif untuk semua model yang telah dilatih.
Menghasilkan: classification report, confusion matrix, ROC curve,
feature importance, SHAP values.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

MODEL_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

CLASS_NAMES = ["Low Addiction", "High Addiction"]


def load_artifacts():
    models = {}
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        path = f"{MODEL_DIR}/{name}.pkl"
        if os.path.exists(path):
            models[name] = joblib.load(path)
    prep = joblib.load(f"{MODEL_DIR}/preprocessing.pkl")
    feature_names = prep["feature_names"]
    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df.drop(columns=["addiction_label"]).values
    y_test = test_df["addiction_label"].values
    return models, feature_names, X_test, y_test


def evaluate_single(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model": name.replace("_", " ").title(),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics, y_pred, y_prob


def plot_confusion_matrices(models, X_test, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
        )
        ax.set_title(name.replace("_", " ").title())
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = f"{REPORTS_DIR}/confusion_matrices.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name.replace('_', ' ').title()} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{REPORTS_DIR}/roc_curves.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_feature_importance(model, feature_names, top_n=20):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return None

    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
    ax.barh(
        range(top_n),
        importances[indices][::-1],
        color=colors[::-1]
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel("Feature Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = f"{REPORTS_DIR}/feature_importance.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_shap_summary(model, X_test, feature_names):
    try:
        import shap
        print("Menghitung SHAP values (sampel 500 data)...")
        sample_idx = np.random.choice(len(X_test), min(500, len(X_test)), replace=False)
        X_sample = X_test[sample_idx]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Jika output adalah list (binary), ambil class 1
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            sv, X_sample,
            feature_names=feature_names,
            show=False, plot_size=None
        )
        path = f"{REPORTS_DIR}/shap_summary.png"
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close("all")
        print(f"Saved: {path}")
        return path
    except Exception as e:
        print(f"SHAP plot gagal: {e}")
        return None


def plot_comparison_bar(results_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["Model"], color=colors[i], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Perbandingan Performa Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar_group_offset in range(len(results_df)):
        for xi, metric in zip(x + bar_group_offset * width, metrics):
            val = results_df.iloc[bar_group_offset][metric]
            ax.text(xi, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = f"{REPORTS_DIR}/model_comparison.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def run_evaluation():
    models, feature_names, X_test, y_test = load_artifacts()

    print("\n=== Evaluasi Model ===")
    results = []
    for name, model in models.items():
        metrics, y_pred, y_prob = evaluate_single(name, model, X_test, y_test)
        results.append(metrics)
        print(f"\n{metrics['Model']}:")
        print(classification_report(y_test, model.predict(X_test),
                                    target_names=CLASS_NAMES))

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{REPORTS_DIR}/model_comparison.csv", index=False)
    print("\nTabel Perbandingan Model:")
    print(results_df.to_string(index=False))

    # Plots
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_comparison_bar(results_df)

    # Feature importance & SHAP untuk XGBoost
    if "xgboost" in models:
        plot_feature_importance(models["xgboost"], feature_names)
        plot_shap_summary(models["xgboost"], X_test, feature_names)

    print("\n=== Evaluasi Selesai ===")
    return results_df


if __name__ == "__main__":
    run_evaluation()
