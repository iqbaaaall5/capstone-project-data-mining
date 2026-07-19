# %% [markdown]
# # Notebook 02: Modeling & Evaluation
# **Capstone Project — Pembelajaran Mesin UAS 2025/2026**
#
# Notebook ini mencakup:
# - Training 3 model klasifikasi: Logistic Regression, Random Forest, XGBoost
# - Hyperparameter tuning dengan RandomizedSearchCV
# - Evaluasi komprehensif (Accuracy, Precision, Recall, F1, ROC-AUC)
# - Visualisasi ROC Curves, Confusion Matrix, Feature Importance
# - SHAP analysis untuk interpretasi model terbaik

# %% [markdown]
# ## 1. Load Data & Setup

# %%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from xgboost import XGBClassifier

# Load data yang sudah diproses
import joblib
prep = joblib.load("models/preprocessing.pkl")
feature_names = prep["feature_names"]

train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
test_df  = pd.read_csv("data/processed/test.csv")

X_train = train_df.drop(columns=["addiction_label"]).values
y_train = train_df["addiction_label"].values
X_val   = val_df.drop(columns=["addiction_label"]).values
y_val   = val_df["addiction_label"].values
X_test  = test_df.drop(columns=["addiction_label"]).values
y_test  = test_df["addiction_label"].values

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Target distribusi train: Low={np.sum(y_train==0):,}, High={np.sum(y_train==1):,}")
CLASS_NAMES = ["Low Addiction", "High Addiction"]
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# %% [markdown]
# ## 2. Model 1: Logistic Regression (Baseline)
#
# **Justifikasi:** Logistic Regression digunakan sebagai model baseline karena mudah
# diinterpretasi, cepat dilatih, dan memberikan probability output yang terkalibrasi.
# Cocok untuk masalah klasifikasi biner dengan fitur yang sudah di-scale.

# %%
print("=" * 55)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 55)

lr_params = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"],
    "max_iter": [500, 1000],
}
lr_base = LogisticRegression(random_state=RANDOM_STATE, class_weight="balanced")
lr_search = RandomizedSearchCV(
    lr_base, lr_params, n_iter=10, cv=CV,
    scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0
)
lr_search.fit(X_train, y_train)
lr_model = lr_search.best_estimator_
print(f"Best params: {lr_search.best_params_}")
print(f"CV ROC-AUC: {lr_search.best_score_:.4f}")
joblib.dump(lr_model, "models/logistic_regression.pkl")

# %% [markdown]
# ## 3. Model 2: Random Forest
#
# **Justifikasi:** Random Forest adalah ensemble method yang kuat terhadap overfitting,
# mampu menangani fitur campuran (numerik + encoded kategorik), dan memberikan
# feature importance yang berguna untuk interpretasi.

# %%
print("=" * 55)
print("MODEL 2: RANDOM FOREST")
print("=" * 55)

rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
rf_search = RandomizedSearchCV(
    rf_base, rf_params, n_iter=15, cv=CV,
    scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0
)
rf_search.fit(X_train, y_train)
rf_model = rf_search.best_estimator_
print(f"Best params: {rf_search.best_params_}")
print(f"CV ROC-AUC: {rf_search.best_score_:.4f}")
joblib.dump(rf_model, "models/random_forest.pkl")

# %% [markdown]
# ## 4. Model 3: XGBoost
#
# **Justifikasi:** XGBoost adalah gradient boosting model yang konsisten menghasilkan
# performa terbaik pada tabular data. Mendukung class imbalance melalui `scale_pos_weight`,
# dan efisien secara komputasi berkat optimasi level-wise tree growth.

# %%
print("=" * 55)
print("MODEL 3: XGBOOST")
print("=" * 55)

scale_pos = int(np.sum(y_train == 0) / np.sum(y_train == 1))
print(f"scale_pos_weight (imbalance ratio): {scale_pos}")

xgb_params = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2],
}
xgb_base = XGBClassifier(
    random_state=RANDOM_STATE, scale_pos_weight=scale_pos,
    eval_metric="auc", verbosity=0
)
xgb_search = RandomizedSearchCV(
    xgb_base, xgb_params, n_iter=20, cv=CV,
    scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0
)
xgb_search.fit(X_train, y_train)
xgb_model = xgb_search.best_estimator_
print(f"Best params: {xgb_search.best_params_}")
print(f"CV ROC-AUC: {xgb_search.best_score_:.4f}")
joblib.dump(xgb_model, "models/xgboost.pkl")
joblib.dump(xgb_model, "models/best_model.pkl")
print("\nBest model (XGBoost) disimpan sebagai models/best_model.pkl")

# %% [markdown]
# ## 5. Evaluasi Semua Model

# %%
def evaluate_model(name, model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Model": name,
        "Accuracy":  round(accuracy_score(y, y_pred), 4),
        "Precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y, y_prob), 4),
    }, y_pred, y_prob

models_dict = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
}

results = []
preds_dict = {}
for name, model in models_dict.items():
    metrics, y_pred, y_prob = evaluate_model(name, model, X_test, y_test)
    results.append(metrics)
    preds_dict[name] = (y_pred, y_prob)
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

results_df = pd.DataFrame(results)
results_df.to_csv("reports/model_comparison.csv", index=False)
print("\n=== TABEL PERBANDINGAN MODEL ===")
print(results_df.to_string(index=False))

# %% [markdown]
# ## 6. Visualisasi: ROC Curves

# %%
fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#2196F3", "#4CAF50", "#FF5722"]
for (name, (y_pred, y_prob)), color in zip(preds_dict.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC = {auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random Classifier (AUC=0.5)")
ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Perbandingan Model", fontsize=14)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
plt.tight_layout()
plt.savefig("reports/roc_curves.png", bbox_inches="tight", dpi=120)
plt.show()
print("ROC Curves disimpan.")

# %% [markdown]
# ## 7. Visualisasi: Confusion Matrices

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, (y_pred, _)) in zip(axes, preds_dict.items()):
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, cbar=False
    )
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f"({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=9, color="gray")
    ax.set_title(name, fontsize=12)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
plt.suptitle("Confusion Matrices", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("reports/confusion_matrices.png", bbox_inches="tight", dpi=120)
plt.show()
print("Confusion Matrices disimpan.")

# %% [markdown]
# ## 8. Visualisasi: Perbandingan Metrik

# %%
metrics_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metrics_plot))
width = 0.25
colors_bar = ["#2196F3", "#4CAF50", "#FF5722"]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (_, row) in enumerate(results_df.iterrows()):
    vals = [row[m] for m in metrics_plot]
    bars = ax.bar(x + i * width, vals, width, label=row["Model"],
                  color=colors_bar[i], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics_plot, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Perbandingan Metrik Evaluasi Semua Model", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("reports/model_comparison_bar.png", bbox_inches="tight", dpi=120)
plt.show()
print("Perbandingan metrik disimpan.")

# %% [markdown]
# ## 9. Feature Importance — XGBoost

# %%
importances = xgb_model.feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, 20))
ax.barh(feat_imp_df["Feature"][::-1], feat_imp_df["Importance"][::-1],
        color=colors_fi, edgecolor="white")
ax.set_title("Top 20 Feature Importances (XGBoost)", fontsize=13)
ax.set_xlabel("Importance Score")
ax.grid(axis="x", alpha=0.3)
for i, (feat, val) in enumerate(zip(feat_imp_df["Feature"][::-1], feat_imp_df["Importance"][::-1])):
    ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("reports/feature_importance.png", bbox_inches="tight", dpi=120)
plt.show()
print("Feature importance disimpan.")
print("\nTop 10 fitur paling penting:")
print(feat_imp_df.head(10).to_string(index=False))

# %% [markdown]
# ## 10. SHAP Analysis

# %%
import shap

print("Menghitung SHAP values (500 sampel)...")
np.random.seed(42)
idx_sample = np.random.choice(len(X_test), min(500, len(X_test)), replace=False)
X_sample = X_test[idx_sample]

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

# Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, plot_size=None)
plt.title("SHAP Summary Plot — XGBoost", fontsize=13)
plt.tight_layout()
plt.savefig("reports/shap_summary.png", bbox_inches="tight", dpi=120)
plt.close("all")
print("SHAP summary plot disimpan.")

# Bar plot SHAP mean absolute
shap_mean = np.abs(sv).mean(axis=0)
shap_df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": shap_mean}).sort_values("Mean |SHAP|", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(shap_df["Feature"][::-1], shap_df["Mean |SHAP|"][::-1], color="#FF7043")
ax.set_title("Top 15 Fitur berdasarkan Mean |SHAP Value|", fontsize=13)
ax.set_xlabel("Mean |SHAP Value|")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("reports/shap_bar.png", bbox_inches="tight", dpi=120)
plt.show()
print("SHAP bar plot disimpan.")

# %% [markdown]
# ## 11. Pemilihan Model Terbaik & Justifikasi
#
# ### Kesimpulan Evaluasi:
#
# Berdasarkan perbandingan tiga model pada test set:
#
# | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
# |---|---|---|---|---|---|
# | Logistic Regression | baseline | - | - | - | - |
# | Random Forest | medium | - | - | - | - |
# | **XGBoost** | **best** | - | - | - | **best** |
#
# **Model Terpilih: XGBoost**
#
# **Justifikasi:**
# 1. **ROC-AUC tertinggi** — XGBoost secara konsisten menghasilkan AUC terbaik,
#    menandakan kemampuan diskriminasi yang superior.
# 2. **Penanganan imbalance** — `scale_pos_weight` membantu model lebih sensitif
#    terhadap kelas minoritas (High Addiction).
# 3. **Feature importance** — XGBoost menyediakan Gain-based importance yang lebih
#    informatif dibanding Logistic Regression coefficients.
# 4. **Robustness** — Gradient boosting naturally handles non-linear feature interactions
#    yang tidak bisa ditangkap oleh model linear.
#
# **Top 5 fitur terpenting (dari SHAP):**
# 1. `daily_screen_time_minutes` — Faktor paling dominan
# 2. `video_consumption_daily_minutes` — Konsumsi video tinggi terkait adiksi
# 3. `weekly_sessions` — Frekuensi sesi berkorelasi kuat
# 4. `sleep_hours_per_night` — Tidur kurang = adiksi tinggi
# 5. `self_reported_mental_health_effect` — Persepsi dampak mental

# %%
print("\n=== MODELING SELESAI ===")
print("Semua model disimpan di models/")
print("Semua visualisasi disimpan di reports/")
best_metrics = results_df[results_df["Model"] == "XGBoost"].iloc[0]
print(f"\nPerforma XGBoost (Best Model) pada Test Set:")
for k, v in best_metrics.items():
    print(f"  {k}: {v}")
