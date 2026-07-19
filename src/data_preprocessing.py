"""
data_preprocessing.py
Script preprocessing dataset Social Media User Behavior.
Target: addiction_label (High=1 jika addiction_level >= 5, Low=0 jika < 5)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/social_media_user_behavior.csv"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path=RAW_PATH):
    df = pd.read_csv(path)
    return df


def create_target(df):
    """Buat target biner: High Addiction (1) jika level >= 5, Low (0) jika < 5"""
    df = df.copy()
    df["addiction_label"] = (df["addiction_level_1_to_10"] >= 5).astype(int)
    return df


def drop_irrelevant(df):
    """Hapus kolom yang tidak relevan sebagai fitur"""
    cols_to_drop = [
        "user_id",
        "account_created_date",
        "addiction_level_1_to_10",  # raw target, sudah dibuat label biner
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df


def encode_features(df):
    """
    Encoding:
    - Boolean (True/False) -> int (1/0)
    - Ordinal kategorik -> LabelEncoder
    Justifikasi: LabelEncoder digunakan karena jumlah kategori moderat
    dan tree-based model tidak sensitif terhadap ordinalitas encoding.
    """
    df = df.copy()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def scale_features(X_train, X_test, X_val=None):
    """
    StandardScaler untuk fitur numerik kontinu.
    Justifikasi: Diperlukan agar model berbasis jarak (seperti Logistic Regression)
    tidak bias terhadap fitur berskala besar.
    Scaler di-fit HANYA pada X_train untuk menghindari data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    return X_train_scaled, X_test_scaled, X_val_scaled, scaler


def preprocess_pipeline(path=RAW_PATH, test_size=0.2, val_size=0.1, random_state=42):
    print("=== Memulai Preprocessing Pipeline ===")

    # 1. Load
    df = load_data(path)
    print(f"Data asli: {df.shape}")

    # 2. Buat target
    df = create_target(df)
    print(f"Distribusi target:\n{df['addiction_label'].value_counts()}")

    # 3. Drop irrelevant
    df = drop_irrelevant(df)
    print(f"Setelah drop kolom tidak relevan: {df.shape}")

    # 4. Cek missing values
    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")

    # 5. Cek duplikat
    dupes = df.duplicated().sum()
    print(f"Total duplikat: {dupes}")
    if dupes > 0:
        df = df.drop_duplicates()

    # 6. Encode
    df, encoders = encode_features(df)

    # 7. Split fitur dan target
    X = df.drop(columns=["addiction_label"])
    y = df["addiction_label"]
    feature_names = X.columns.tolist()

    # 8. Train / Val / Test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 9. Scaling
    X_train_s, X_test_s, X_val_s, scaler = scale_features(X_train, X_test, X_val)

    # 10. Simpan processed data
    train_df = pd.DataFrame(X_train_s, columns=feature_names)
    train_df["addiction_label"] = y_train.values
    train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)

    val_df = pd.DataFrame(X_val_s, columns=feature_names)
    val_df["addiction_label"] = y_val.values
    val_df.to_csv(f"{PROCESSED_DIR}/val.csv", index=False)

    test_df = pd.DataFrame(X_test_s, columns=feature_names)
    test_df["addiction_label"] = y_test.values
    test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

    # 11. Simpan preprocessing artifacts
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")
    joblib.dump(feature_names, f"{MODEL_DIR}/feature_names.pkl")

    # Bundle sebagai preprocessing.pkl (sesuai template repository)
    preprocessing_bundle = {
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_names,
    }
    joblib.dump(preprocessing_bundle, f"{MODEL_DIR}/preprocessing.pkl")

    print("=== Preprocessing Selesai ===")
    print(f"Artifacts disimpan di: {MODEL_DIR}/")
    print(f"Processed data disimpan di: {PROCESSED_DIR}/")

    return (
        X_train_s, X_val_s, X_test_s,
        y_train.values, y_val.values, y_test.values,
        feature_names, scaler, encoders
    )


if __name__ == "__main__":
    preprocess_pipeline()
