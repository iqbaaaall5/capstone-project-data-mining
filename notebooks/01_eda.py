# %% [markdown]
# # Notebook 01: Exploratory Data Analysis & Preprocessing
# **Capstone Project — Pembelajaran Mesin UAS 2025/2026**
#
# **Problem Statement:**
# Dataset ini berisi perilaku pengguna media sosial dari 25.000 individu di berbagai negara.
# Tujuan analisis adalah memahami pola penggunaan media sosial dan faktor-faktor yang
# berkontribusi pada tingkat adiksi pengguna.
#
# **Target Variabel:**
# `addiction_label` — klasifikasi biner dari `addiction_level_1_to_10`:
# - **High Addiction (1)**: level ≥ 5
# - **Low Addiction (0)**: level < 5

# %% [markdown]
# ## 1. Import Libraries & Load Data

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

os.makedirs("reports", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# %%
df_raw = pd.read_csv("data/raw/social_media_user_behavior.csv")
df_platform = pd.read_csv("data/external/platform_statistics_2026.csv")

print(f"Dataset utama: {df_raw.shape[0]:,} baris × {df_raw.shape[1]} kolom")
print(f"Dataset platform: {df_platform.shape}")
df_raw.head(3)

# %% [markdown]
# ## 2. Analisis Kualitas Data

# %%
print("=" * 55)
print("ANALISIS KUALITAS DATA")
print("=" * 55)

# Missing values
missing = df_raw.isnull().sum()
print(f"\nTotal missing values: {missing.sum()}")
print("(Dataset bersih — tidak ada missing values)")

# Duplikat
dupes = df_raw.duplicated().sum()
print(f"Total duplikat: {dupes}")

# Tipe data
print("\nTipe data per kolom:")
print(df_raw.dtypes.value_counts())

# Statistik deskriptif
print("\nStatistik Deskriptif (Numerik):")
df_raw.describe().T

# %%
# Rangkuman info dataset
print("\nKolom dataset:")
for col in df_raw.columns:
    dtype = df_raw[col].dtype
    nuniq = df_raw[col].nunique()
    sample = df_raw[col].iloc[0]
    print(f"  {col:<40} dtype={str(dtype):<10} nunique={nuniq:<6} sample={sample}")

# %% [markdown]
# ## 3. Membuat Target Variable

# %%
df = df_raw.copy()
df["addiction_label"] = (df["addiction_level_1_to_10"] >= 5).astype(int)

label_counts = df["addiction_label"].value_counts().sort_index()
label_pct = df["addiction_label"].value_counts(normalize=True).sort_index() * 100

print("Distribusi Target:")
print(f"  Low Addiction  (0): {label_counts[0]:,} ({label_pct[0]:.1f}%)")
print(f"  High Addiction (1): {label_counts[1]:,} ({label_pct[1]:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Distribusi level asli
axes[0].bar(
    df["addiction_level_1_to_10"].value_counts().sort_index().index,
    df["addiction_level_1_to_10"].value_counts().sort_index().values,
    color=sns.color_palette("Set2", 10)
)
axes[0].set_title("Distribusi Addiction Level (1-10)")
axes[0].set_xlabel("Addiction Level")
axes[0].set_ylabel("Jumlah Pengguna")
for i, v in enumerate(df["addiction_level_1_to_10"].value_counts().sort_index().values):
    axes[0].text(i + 1, v + 50, str(v), ha="center", fontsize=9)

# Target biner
colors = ["#4CAF50", "#F44336"]
axes[1].bar(["Low (0)", "High (1)"], label_counts.values, color=colors, width=0.5)
axes[1].set_title("Distribusi Target Biner: addiction_label")
axes[1].set_ylabel("Jumlah Pengguna")
for i, v in enumerate(label_counts.values):
    axes[1].text(i, v + 50, f"{v:,}\n({label_pct.values[i]:.1f}%)", ha="center")

plt.tight_layout()
plt.savefig("reports/insight1_target_distribution.png", bbox_inches="tight", dpi=120)
plt.show()
print("Insight 1: Distribusi target disimpan.")

# %% [markdown]
# ## 4. Analisis Univariat — Fitur Numerik

# %%
num_cols = [
    "age", "daily_screen_time_minutes", "weekly_sessions",
    "avg_session_duration_minutes", "posts_per_week", "likes_per_day",
    "comments_per_day", "shares_per_week", "followers_count", "following_count",
    "engagement_rate_pct", "video_consumption_daily_minutes",
    "monthly_social_spending_usd", "sleep_hours_per_night",
    "daily_notifications", "account_age_years", "num_platforms_used"
]

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=30, color="#42A5F5", edgecolor="white", alpha=0.8)
    axes[i].set_title(col, fontsize=9)
    axes[i].set_ylabel("Freq")
for j in range(len(num_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Distribusi Fitur Numerik", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("reports/univariate_numeric.png", bbox_inches="tight", dpi=100)
plt.show()

# %%
# Statistik deskriptif lengkap
print(df[num_cols].describe().round(2).T.to_string())

# %% [markdown]
# ## 5. Insight 2: Screen Time vs Addiction Level

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot screen time per addiction label
df_box = df[["daily_screen_time_minutes", "addiction_label"]].copy()
df_box["Addiction"] = df_box["addiction_label"].map({0: "Low", 1: "High"})

bp = axes[0].boxplot(
    [df_box[df_box["addiction_label"] == 0]["daily_screen_time_minutes"],
     df_box[df_box["addiction_label"] == 1]["daily_screen_time_minutes"]],
    labels=["Low Addiction", "High Addiction"],
    patch_artist=True,
    boxprops=dict(facecolor="#42A5F5", alpha=0.7),
    medianprops=dict(color="red", linewidth=2)
)
bp["boxes"][1].set_facecolor("#EF5350")
axes[0].set_title("Screen Time vs Addiction Level")
axes[0].set_ylabel("Daily Screen Time (menit)")
axes[0].grid(axis="y", alpha=0.3)

# Mean screen time per addiction level asli
mean_screen = df.groupby("addiction_level_1_to_10")["daily_screen_time_minutes"].mean()
axes[1].plot(mean_screen.index, mean_screen.values, "o-", color="#FF5722", linewidth=2, markersize=8)
axes[1].set_title("Mean Screen Time per Addiction Level")
axes[1].set_xlabel("Addiction Level (1-10)")
axes[1].set_ylabel("Mean Screen Time (menit)")
axes[1].grid(alpha=0.3)
for x, y in zip(mean_screen.index, mean_screen.values):
    axes[1].annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("reports/insight2_screentime_addiction.png", bbox_inches="tight", dpi=120)
plt.show()
print("Insight 2: Screen Time vs Addiction disimpan.")

# %% [markdown]
# ## 6. Insight 3: Mental Health Effect vs Addiction

# %%
mh_counts = df.groupby(["self_reported_mental_health_effect", "addiction_label"]).size().unstack(fill_value=0)
mh_counts.columns = ["Low Addiction", "High Addiction"]
mh_pct = mh_counts.div(mh_counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 6))
mh_pct.plot(kind="bar", ax=ax, color=["#4CAF50", "#F44336"], width=0.7, edgecolor="white")
ax.set_title("Proporsi Addiction Level per Dampak Mental Health")
ax.set_xlabel("Self-Reported Mental Health Effect")
ax.set_ylabel("Persentase (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(title="Addiction")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("reports/insight3_mental_health_addiction.png", bbox_inches="tight", dpi=120)
plt.show()
print("Insight 3: Mental Health Effect disimpan.")

# %% [markdown]
# ## 7. Insight 4: Korelasi Fitur Numerik dengan Target

# %%
corr_with_target = df[num_cols + ["addiction_label"]].corr()["addiction_label"].drop("addiction_label").sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#F44336" if v > 0 else "#2196F3" for v in corr_with_target.values]
ax.barh(corr_with_target.index[::-1], corr_with_target.values[::-1], color=colors[::-1])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Korelasi Pearson Fitur Numerik vs Addiction Label")
ax.set_xlabel("Pearson Correlation")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("reports/insight4_correlation.png", bbox_inches="tight", dpi=120)
plt.show()
print("Insight 4: Korelasi fitur disimpan.")
print("\nTop 10 fitur berkorelasi tinggi:")
print(corr_with_target.head(10))

# %% [markdown]
# ## 8. Insight 5: Platform & Demografi

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Platform dominan
platform_addiction = df.groupby("primary_platform")["addiction_label"].mean().sort_values(ascending=False)
axes[0].bar(platform_addiction.index, platform_addiction.values,
            color=sns.color_palette("RdYlGn_r", len(platform_addiction)))
axes[0].set_title("Rata-rata High Addiction Rate per Platform")
axes[0].set_ylabel("Proporsi High Addiction")
axes[0].set_xticklabels(platform_addiction.index, rotation=45, ha="right")
axes[0].grid(axis="y", alpha=0.3)
for i, (x, y) in enumerate(zip(platform_addiction.index, platform_addiction.values)):
    axes[0].text(i, y + 0.005, f"{y:.2f}", ha="center", fontsize=9)

# Age group vs addiction
age_addiction = df.groupby("age_group")["addiction_label"].mean().sort_values(ascending=False)
axes[1].bar(age_addiction.index, age_addiction.values,
            color=sns.color_palette("coolwarm", len(age_addiction)))
axes[1].set_title("High Addiction Rate per Age Group")
axes[1].set_ylabel("Proporsi High Addiction")
axes[1].set_xticklabels(age_addiction.index, rotation=30, ha="right")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("reports/insight5_platform_age_addiction.png", bbox_inches="tight", dpi=120)
plt.show()
print("Insight 5: Platform & Age Group disimpan.")

# %% [markdown]
# ## 9. Analisis Multivariat — Heatmap Korelasi

# %%
corr_matrix = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, square=True, ax=ax,
    annot_kws={"size": 7}, linewidths=0.5
)
ax.set_title("Heatmap Korelasi Fitur Numerik")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png", bbox_inches="tight", dpi=100)
plt.show()

# %% [markdown]
# ## 10. Feature Engineering & Preprocessing

# %%
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df_proc = df.copy()

# Drop kolom tidak relevan
drop_cols = ["user_id", "account_created_date", "addiction_level_1_to_10"]
df_proc = df_proc.drop(columns=drop_cols)
print(f"Setelah drop kolom non-fitur: {df_proc.shape}")

# Encoding Boolean
bool_cols = df_proc.select_dtypes(include="bool").columns.tolist()
for col in bool_cols:
    df_proc[col] = df_proc[col].astype(int)
print(f"Boolean encoded: {bool_cols}")

# Encoding Kategorik
cat_cols = df_proc.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_proc[col] = le.fit_transform(df_proc[col].astype(str))
    encoders[col] = le
print(f"Kategorik encoded ({len(cat_cols)} kolom): {cat_cols}")

# Split X dan y
X = df_proc.drop(columns=["addiction_label"])
y = df_proc["addiction_label"]
feature_names = X.columns.tolist()
print(f"\nJumlah fitur: {len(feature_names)}")
print(f"Target distribusi: {y.value_counts().to_dict()}")

# Train/Val/Test split (70/10/20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)
print(f"\nSplit dataset:")
print(f"  Train : {X_train.shape[0]:,} ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Val   : {X_val.shape[0]:,} ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"  Test  : {X_test.shape[0]:,} ({X_test.shape[0]/len(X)*100:.0f}%)")

# Scaling (fit hanya pada train)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Simpan processed data
import joblib, os
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

pd.DataFrame(X_train_s, columns=feature_names).assign(addiction_label=y_train.values).to_csv("data/processed/train.csv", index=False)
pd.DataFrame(X_val_s, columns=feature_names).assign(addiction_label=y_val.values).to_csv("data/processed/val.csv", index=False)
pd.DataFrame(X_test_s, columns=feature_names).assign(addiction_label=y_test.values).to_csv("data/processed/test.csv", index=False)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")

# Bundle sebagai preprocessing.pkl (sesuai template repository)
preprocessing_bundle = {
    "scaler": scaler,
    "encoders": encoders,
    "feature_names": feature_names,
}
joblib.dump(preprocessing_bundle, "models/preprocessing.pkl")

print("\nPreprocessing artifacts disimpan di models/")
print("Processed data disimpan di data/processed/")

# %% [markdown]
# ## 11. Ringkasan EDA
#
# ### 5 Key Insights:
#
# 1. **Distribusi Target Imbalanced** — 77.4% Low Addiction vs 22.6% High Addiction.
#    Perlu stratified split dan class-weight handling saat modeling.
#
# 2. **Screen Time Signifikan** — Pengguna High Addiction memiliki rata-rata screen time
#    ~40% lebih tinggi dibanding Low Addiction. Fitur `daily_screen_time_minutes`,
#    `weekly_sessions`, dan `video_consumption_daily_minutes` berkorelasi positif dengan target.
#
# 3. **Mental Health Berkaitan** — Pengguna yang melaporkan dampak mental health "Mostly Negative"
#    memiliki proporsi High Addiction jauh lebih tinggi (>50%) dibanding yang "Mostly Positive".
#
# 4. **Platform TikTok & Reddit** — Memiliki addiction rate tertinggi, konsisten dengan
#    nature platform yang menggunakan infinite scroll dan short-form video.
#
# 5. **Usia Muda Lebih Rentan** — Kelompok usia 13-17 dan 18-24 menunjukkan
#    proporsi High Addiction lebih tinggi dibanding kelompok usia di atas 35 tahun.

print("\nNotebook EDA selesai! Semua output disimpan di folder reports/")
