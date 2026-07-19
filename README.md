# 📱 Social Media Addiction Predictor
**Capstone Project — UAS Pembelajaran Mesin 2025/2026**

Prediksi tingkat adiksi media sosial pengguna menggunakan Machine Learning berbasis data perilaku 25.000 pengguna dari berbagai negara.

---

## 🎯 Problem Statement

Adiksi media sosial adalah masalah kesehatan mental yang berkembang pesat. Proyek ini membangun model klasifikasi untuk mengidentifikasi pengguna dengan risiko **High Addiction** (addiction level ≥ 5 dari skala 1–10) berdasarkan 44 atribut perilaku penggunaan.

**Target:** `addiction_label` — `1` (High Addiction) atau `0` (Low Addiction)

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
```bash
pip install -r requirements.txt
```

### 2. Jalankan Notebook EDA (training otomatis tersimpan)
```bash
python notebooks/01_eda.py
python notebooks/02_modeling.py
python notebooks/03_interpretation.py
```

### 3. Atau gunakan src scripts langsung
```bash
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate_model.py
```

### 4. Jalankan Streamlit App
```bash
streamlit run app/app.py
```

---

## 📊 Hasil Model

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.78 | ~0.62 | ~0.71 | ~0.66 | ~0.84 |
| Random Forest | ~0.82 | ~0.71 | ~0.72 | ~0.71 | ~0.89 |
| **XGBoost** ⭐ | **~0.84** | **~0.74** | **~0.76** | **~0.75** | **~0.91** |

**Model terbaik: XGBoost** — dipilih berdasarkan ROC-AUC tertinggi dan F1-Score terbaik.

---

## 🔍 5 Key Insights

1. **Screen time > 3 jam/hari** adalah prediktor terkuat High Addiction
2. **Konsumsi video** (TikTok/Reels) berkontribusi besar pada adiksi
3. **Usia 13–24 tahun** paling rentan terhadap adiksi media sosial
4. **Mental health negatif** berkorelasi kuat dengan adiksi tinggi
5. **Hanya 18% pengguna** mengaktifkan screen time limits

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **ML:** Scikit-learn, XGBoost
- **Interpretation:** SHAP
- **Deployment:** Streamlit

---

## 👤 Author

Capstone Project — Mata Kuliah Pembelajaran Mesin  
Universitas Dian Nuswantoro (UDINUS) — Semester Genap 2025/2026
