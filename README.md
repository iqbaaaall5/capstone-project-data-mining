# 📱 Social Media Addiction Predictor

**Capstone Project — UAS Pembelajaran Mesin 2025/2026**

Prediksi tingkat adiksi media sosial pengguna menggunakan Machine Learning berbasis data perilaku 25.000 pengguna dari berbagai negara.

---

## 🎯 Problem Statement

Adiksi media sosial adalah masalah kesehatan mental yang berkembang pesat. Proyek ini membangun model klasifikasi untuk mengidentifikasi pengguna dengan risiko **High Addiction** (addiction level ≥ 5 dari skala 1–10) berdasarkan 44 atribut perilaku penggunaan.

**Target:** `addiction_label` — `1` (High Addiction) atau `0` (Low Addiction)

---

## 🌐 Live Demo

**Aplikasi:** https://capstone-project-data-mining-rsfsdvoappbvycj7dsik7g8.streamlit.app/

**Repository:** https://github.com/iqbaaaall5/capstone-project-data-mining

---

## 📂 Struktur Repositori

```
capstone-project-data-mining/
├── data/
│   ├── social_media_user_behavior.csv   # Dataset utama (25.000 baris)
│   ├── platform_statistics_2026.csv     # Data statistik platform
│   ├── raw/                             # Data mentah backup
│   └── processed/                       # Data setelah preprocessing
├── notebooks/
│   ├── 01_eda.py                        # EDA & Preprocessing
│   ├── 02_modeling.py                   # Training & Evaluasi model
│   └── 03_interpretation.py            # SHAP & interpretasi
├── src/
│   ├── data_preprocessing.py           # Pipeline preprocessing
│   ├── train_model.py                  # Script training
│   ├── evaluate_model.py               # Script evaluasi
│   └── utils.py                        # Fungsi utilitas
├── models/
│   ├── best_model.pkl                  # XGBoost (best model)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
├── app/
│   └── app.py                          # Streamlit application
├── reports/                            # Visualisasi & laporan
├── requirements.txt
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Jalankan Notebook EDA & Modeling (urutan wajib)
```
python notebooks/01_eda.py
python notebooks/02_modeling.py
python notebooks/03_interpretation.py
```

### 3. Jalankan Streamlit App
```
streamlit run app/app.py
```

---

## 📊 Hasil Model

Evaluasi dilakukan pada test set (20% data, belum pernah dilihat model selama training/tuning). Karena distribusi target imbalanced (77% Low Addiction vs 23% High Addiction), F1-Score dan ROC-AUC menjadi metrik acuan utama, bukan hanya accuracy.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8150 | 0.4964 | 0.8359 | 0.6229 | 0.9081 |
| Random Forest | 0.8540 | 0.5764 | 0.7593 | 0.6553 | 0.9117 |
| **XGBoost** ⭐ | 0.8394 | 0.5397 | **0.8260** | 0.6528 | **0.9136** |

*(Angka di atas diambil langsung dari `reports/model_comparison.csv`)*

**Model terbaik: XGBoost** — Random Forest sebenarnya sedikit lebih tinggi di accuracy dan F1-score, tetapi XGBoost dipilih karena:
1. **ROC-AUC tertinggi** (0.9136) — kemampuan diskriminasi antar kelas paling baik secara keseluruhan
2. **Recall jauh lebih tinggi** (0.8260 vs 0.7593 milik Random Forest) — penting karena konteks proyek ini adalah deteksi dini, di mana melewatkan kasus High Addiction (False Negative) memiliki konsekuensi lebih besar dibanding kesalahan sebaliknya
3. **scale_pos_weight** membantu model lebih sensitif terhadap kelas minoritas (High Addiction)
4. Feature importance berbasis Gain yang lebih informatif, dan kompatibel dengan analisis SHAP untuk interpretasi

---

## 🔍 5 Key Insights

1. **Screen time** adalah prediktor terkuat — pengguna High Addiction memiliki screen time harian ~40% lebih tinggi
2. **Konsumsi video** (TikTok/Reels/short-form content) berkontribusi besar pada adiksi
3. **Usia 13–24 tahun** menunjukkan proporsi High Addiction lebih besar dibanding usia di atas 35 tahun
4. **Mental health negatif** berkorelasi kuat dengan adiksi tinggi (>50% pada kelompok "Mostly Negative")
5. **Jam tidur lebih pendek** berasosiasi dengan skor kecanduan lebih tinggi

Top 5 fitur terpenting (dari SHAP): `daily_screen_time_minutes`, `video_consumption_daily_minutes`, `weekly_sessions`, `sleep_hours_per_night`, `self_reported_mental_health_effect`.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **ML:** Scikit-learn, XGBoost
- **Interpretation:** SHAP
- **Deployment:** Streamlit Community Cloud

---

## ⚠️ Keterbatasan

- Target bersifat self-report sehingga berpotensi bias subjektif
- Precision pada kelas minoritas (High Addiction) masih relatif rendah (0.50–0.58), artinya cukup banyak False Positive
- Dataset bersifat cross-sectional (satu titik waktu), belum menangkap perubahan pola pengguna dari waktu ke waktu

---

## 👤 Author

**Muhammad Iqbal** — A11.2024.15747
Capstone Project — Mata Kuliah Pembelajaran Mesin
Universitas Dian Nuswantoro (UDINUS) — Semester Genap 2025/2026
