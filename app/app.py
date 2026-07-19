"""
app.py — Social Media Addiction Predictor
Capstone Project Pembelajaran Mesin UAS 2025/2026
Streamlit multi-page application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# ── path setup ───────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="Social Media Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ───────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    return pd.read_csv(os.path.join(ROOT, "data", "raw", "social_media_user_behavior.csv"))

@st.cache_data
def load_platform_data():
    return pd.read_csv(os.path.join(ROOT, "data", "external", "platform_statistics_2026.csv"))

@st.cache_resource
def load_model_artifacts():
    model_dir = os.path.join(ROOT, "models")
    best    = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    # Load preprocessing bundle (preprocessing.pkl sesuai template)
    prep    = joblib.load(os.path.join(model_dir, "preprocessing.pkl"))
    scaler  = prep["scaler"]
    enc     = prep["encoders"]
    feats   = prep["feature_names"]
    models  = {
        "Logistic Regression": joblib.load(os.path.join(model_dir, "logistic_regression.pkl")),
        "Random Forest":       joblib.load(os.path.join(model_dir, "random_forest.pkl")),
        "XGBoost":             joblib.load(os.path.join(model_dir, "xgboost.pkl")),
    }
    return best, scaler, enc, feats, models

# ── sidebar navigation ────────────────────────────────────────
st.sidebar.title("📱 Addiction Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda", "📊 Dashboard EDA", "🤖 Demo Model", "📈 Evaluasi Model", "💡 Interpretasi", "📚 Dokumentasi"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Capstone Project — UAS Pembelajaran Mesin 2025/2026")

# ── check if models exist ────────────────────────────────────
models_ready = os.path.exists(os.path.join(ROOT, "models", "best_model.pkl"))

# ══════════════════════════════════════════════════════════════
# PAGE: BERANDA
# ══════════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.title("📱 Social Media Addiction Predictor")
    st.markdown("### Capstone Project — Pembelajaran Mesin UAS 2025/2026")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    df = load_raw_data()
    with col1:
        st.metric("Total Data", f"{len(df):,}")
    with col2:
        st.metric("Jumlah Fitur", f"{df.shape[1] - 1}")
    with col3:
        high = int((df["addiction_level_1_to_10"] >= 5).sum())
        st.metric("High Addiction", f"{high:,} ({high/len(df)*100:.1f}%)")
    with col4:
        st.metric("Platform", df["primary_platform"].nunique())

    st.markdown("---")
    st.subheader("📌 Tentang Proyek")
    st.markdown("""
Proyek ini mengembangkan **model machine learning untuk memprediksi tingkat adiksi media sosial**
berdasarkan perilaku dan karakteristik pengguna.

**Problem Statement:**
Di era digital, adiksi media sosial menjadi masalah kesehatan mental yang serius. Dataset ini
mencakup 25.000 pengguna dari berbagai negara dengan 44 atribut perilaku penggunaan media sosial.

**Target:**
- `High Addiction` (1): addiction_level ≥ 5
- `Low Addiction` (0): addiction_level < 5

**Pipeline ML:**
1. EDA & Preprocessing
2. Training 3 model: Logistic Regression, Random Forest, XGBoost
3. Hyperparameter tuning (RandomizedSearchCV)
4. Evaluasi komprehensif
5. SHAP interpretability
    """)

    st.subheader("🗂️ Struktur Navigasi")
    st.markdown("""
| Halaman | Deskripsi |
|---|---|
| 📊 Dashboard EDA | Visualisasi interaktif analisis data |
| 🤖 Demo Model | Input data → Prediksi real-time |
| 📈 Evaluasi Model | Metrik & visualisasi performa |
| 💡 Interpretasi | Feature importance & SHAP |
| 📚 Dokumentasi | Dataset & metodologi |
    """)

# ══════════════════════════════════════════════════════════════
# PAGE: DASHBOARD EDA
# ══════════════════════════════════════════════════════════════
elif page == "📊 Dashboard EDA":
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("📊 Dashboard EDA")
    st.markdown("Eksplorasi interaktif dataset Social Media User Behavior")
    df = load_raw_data()
    df["addiction_label"] = (df["addiction_level_1_to_10"] >= 5).astype(int)
    df["Addiction"] = df["addiction_label"].map({0: "Low", 1: "High"})

    tab1, tab2, tab3 = st.tabs(["📌 Distribusi", "🔗 Korelasi", "🌍 Demografi"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="addiction_level_1_to_10",
                               title="Distribusi Addiction Level",
                               color_discrete_sequence=["#42A5F5"],
                               labels={"addiction_level_1_to_10": "Addiction Level"})
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            counts = df["Addiction"].value_counts()
            fig = px.pie(values=counts.values, names=counts.index,
                         title="Proporsi High vs Low Addiction",
                         color_discrete_map={"Low": "#4CAF50", "High": "#F44336"})
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.box(df, x="Addiction", y="daily_screen_time_minutes",
                         color="Addiction",
                         color_discrete_map={"Low": "#4CAF50", "High": "#F44336"},
                         title="Screen Time vs Addiction Level")
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = px.box(df, x="Addiction", y="sleep_hours_per_night",
                         color="Addiction",
                         color_discrete_map={"Low": "#4CAF50", "High": "#F44336"},
                         title="Sleep Hours vs Addiction Level")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_cols = ["age", "daily_screen_time_minutes", "weekly_sessions",
                    "sleep_hours_per_night", "video_consumption_daily_minutes",
                    "posts_per_week", "likes_per_day", "engagement_rate_pct",
                    "daily_notifications", "addiction_level_1_to_10"]
        corr = df[num_cols].corr().round(3)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        title="Heatmap Korelasi Fitur Numerik", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

        corr_target = corr["addiction_level_1_to_10"].drop("addiction_level_1_to_10").sort_values(key=abs, ascending=False)
        fig2 = px.bar(x=corr_target.index, y=corr_target.values,
                      color=corr_target.values, color_continuous_scale="RdBu_r",
                      title="Korelasi Fitur vs Addiction Level",
                      labels={"x": "Fitur", "y": "Pearson r"})
        fig2.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            plat = df.groupby("primary_platform")["addiction_label"].mean().reset_index()
            plat.columns = ["Platform", "High Addiction Rate"]
            plat = plat.sort_values("High Addiction Rate", ascending=False)
            fig = px.bar(plat, x="Platform", y="High Addiction Rate",
                         color="High Addiction Rate", color_continuous_scale="Reds",
                         title="High Addiction Rate per Platform")
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            age = df.groupby("age_group")["addiction_label"].mean().reset_index()
            age.columns = ["Age Group", "High Addiction Rate"]
            fig = px.bar(age.sort_values("High Addiction Rate", ascending=False),
                         x="Age Group", y="High Addiction Rate",
                         color="High Addiction Rate", color_continuous_scale="Oranges",
                         title="High Addiction Rate per Age Group")
            st.plotly_chart(fig, use_container_width=True)

        mh = df.groupby("self_reported_mental_health_effect")["addiction_label"].mean().reset_index()
        mh.columns = ["Mental Health Effect", "High Addiction Rate"]
        fig = px.bar(mh.sort_values("High Addiction Rate", ascending=False),
                     x="Mental Health Effect", y="High Addiction Rate",
                     color="High Addiction Rate", color_continuous_scale="RdYlGn_r",
                     title="High Addiction Rate berdasarkan Dampak Mental Health")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: DEMO MODEL
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Demo Model":
    st.title("🤖 Demo Prediksi Adiksi Media Sosial")
    st.markdown("Masukkan data pengguna untuk mendapatkan prediksi tingkat adiksi.")

    if not models_ready:
        st.warning("⚠️ Model belum tersedia. Jalankan notebook training terlebih dahulu.")
        st.stop()

    best_model, scaler, encoders, feature_names, _ = load_model_artifacts()
    df_raw = load_raw_data()

    st.markdown("### Input Data Pengguna")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Profil Pengguna**")
        age = st.slider("Usia", 13, 80, 25)
        gender = st.selectbox("Gender", sorted(df_raw["gender"].unique()))
        country = st.selectbox("Negara", sorted(df_raw["country"].unique()))
        occupation = st.selectbox("Pekerjaan", sorted(df_raw["occupation"].unique()))
        education = st.selectbox("Pendidikan", sorted(df_raw["education_level"].unique()))
        income = st.selectbox("Pendapatan", sorted(df_raw["income_bracket"].unique()))
        relationship = st.selectbox("Status Hubungan", sorted(df_raw["relationship_status"].unique()))

    with col2:
        st.markdown("**📱 Perilaku Penggunaan**")
        platform = st.selectbox("Platform Utama", sorted(df_raw["primary_platform"].unique()))
        num_platforms = st.slider("Jumlah Platform Digunakan", 1, 15, 4)
        screen_time = st.slider("Screen Time Harian (menit)", 0, 720, 120)
        weekly_sessions = st.slider("Sesi per Minggu", 1, 50, 14)
        avg_session = st.slider("Rata-rata Durasi Sesi (menit)", 5, 300, 60)
        content_type = st.selectbox("Tipe Konten Favorit", sorted(df_raw["preferred_content_type"].unique()))
        device = st.selectbox("Perangkat Utama", sorted(df_raw["primary_device"].unique()))
        purpose = st.selectbox("Tujuan Penggunaan", sorted(df_raw["usage_purpose"].unique()))

    with col3:
        st.markdown("**📊 Aktivitas & Dampak**")
        posts_week = st.slider("Post per Minggu", 0, 50, 2)
        likes_day = st.slider("Like per Hari", 0, 200, 15)
        comments_day = st.slider("Komentar per Hari", 0, 50, 3)
        shares_week = st.slider("Share per Minggu", 0, 50, 2)
        video_daily = st.slider("Konsumsi Video Harian (menit)", 0, 500, 60)
        sleep_hours = st.slider("Jam Tidur per Malam", 3.0, 12.0, 7.0, 0.1)
        notifications = st.slider("Notifikasi Harian", 0, 300, 30)
        mental_health = st.selectbox("Dampak Mental Health", sorted(df_raw["self_reported_mental_health_effect"].unique()))
        productivity = st.selectbox("Dampak Produktivitas", sorted(df_raw["productivity_impact"].unique()))
        satisfaction = st.selectbox("Kepuasan Platform", sorted(df_raw["platform_satisfaction"].unique()))

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        followers = st.number_input("Jumlah Followers", 0, 10_000_000, 500)
        following = st.number_input("Jumlah Following", 0, 10_000_000, 300)
        engagement = st.number_input("Engagement Rate (%)", 0.0, 100.0, 2.0, 0.1)
        monthly_spend = st.number_input("Pengeluaran Sosmed/Bulan (USD)", 0.0, 1000.0, 10.0, 0.5)
        ad_freq = st.selectbox("Frekuensi Klik Iklan", sorted(df_raw["ad_click_frequency"].unique()))
        acc_age = st.slider("Usia Akun (tahun)", 0.0, 20.0, 3.0, 0.1)
    with col_b:
        purchased = st.checkbox("Pernah Beli via Social Media")
        follows_inf = st.checkbox("Mengikuti Influencer")
        uses_privacy = st.checkbox("Menggunakan Privacy Settings")
        experienced_cb = st.checkbox("Pernah Mengalami Cyberbullying")
        report_fake = st.selectbox("Frekuensi Lapor Fake News", sorted(df_raw["reports_fake_news_frequency"].unique()))
        is_verified = st.checkbox("Akun Terverifikasi")
        is_creator = st.checkbox("Content Creator")
        uses_ai = st.checkbox("Menggunakan Fitur AI")
        checks_phone = st.checkbox("Cek HP Pertama Kali Pagi")
        screen_limits = st.checkbox("Menggunakan Screen Time Limits")

    age_group_map = {
        range(13, 18): "13-17", range(18, 25): "18-24",
        range(25, 35): "25-34", range(35, 45): "35-44",
        range(45, 55): "45-54", range(55, 65): "55-64", range(65, 81): "65+"
    }
    age_group = "25-34"
    for r, g in age_group_map.items():
        if age in r:
            age_group = g; break

    input_dict = {
        "age": age, "age_group": age_group, "gender": gender, "country": country,
        "occupation": occupation, "education_level": education, "income_bracket": income,
        "relationship_status": relationship, "primary_platform": platform,
        "num_platforms_used": num_platforms, "daily_screen_time_minutes": screen_time,
        "weekly_sessions": weekly_sessions, "avg_session_duration_minutes": avg_session,
        "preferred_content_type": content_type, "primary_device": device,
        "usage_purpose": purpose, "posts_per_week": posts_week, "likes_per_day": likes_day,
        "comments_per_day": comments_day, "shares_per_week": shares_week,
        "followers_count": followers, "following_count": following,
        "engagement_rate_pct": engagement, "video_consumption_daily_minutes": video_daily,
        "has_purchased_via_social": purchased, "follows_influencers": follows_inf,
        "ad_click_frequency": ad_freq, "monthly_social_spending_usd": monthly_spend,
        "uses_privacy_settings": uses_privacy, "experienced_cyberbullying": experienced_cb,
        "reports_fake_news_frequency": report_fake, "self_reported_mental_health_effect": mental_health,
        "sleep_hours_per_night": sleep_hours, "productivity_impact": productivity,
        "platform_satisfaction": satisfaction, "account_age_years": acc_age,
        "is_verified_account": is_verified, "is_content_creator": is_creator,
        "uses_ai_features": uses_ai, "daily_notifications": notifications,
        "checks_phone_first_morning": checks_phone, "uses_screen_time_limits": screen_limits,
    }

    if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_dict])
        for col in input_df.select_dtypes(include="bool").columns:
            input_df[col] = input_df[col].astype(int)
        for col, le in encoders.items():
            if col in input_df.columns:
                val = str(input_df[col].iloc[0])
                if val in le.classes_:
                    input_df[col] = le.transform([val])
                else:
                    input_df[col] = 0
        X_input = input_df[feature_names].values
        X_scaled = scaler.transform(X_input)
        pred = best_model.predict(X_scaled)[0]
        prob = best_model.predict_proba(X_scaled)[0]

        st.markdown("---")
        st.markdown("## 🎯 Hasil Prediksi")
        r1, r2, r3 = st.columns(3)
        with r1:
            label = "🔴 HIGH ADDICTION" if pred == 1 else "🟢 LOW ADDICTION"
            st.metric("Prediksi", label)
        with r2:
            st.metric("Probabilitas High Addiction", f"{prob[1]*100:.1f}%")
        with r3:
            st.metric("Probabilitas Low Addiction", f"{prob[0]*100:.1f}%")

        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob[1]*100,
            title={"text": "Skor Risiko Adiksi (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#F44336" if pred == 1 else "#4CAF50"},
                   "steps": [{"range": [0, 40], "color": "#E8F5E9"},
                              {"range": [40, 70], "color": "#FFF9C4"},
                              {"range": [70, 100], "color": "#FFEBEE"}],
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        if pred == 1:
            st.error("⚠️ **Perhatian:** Profil ini menunjukkan risiko tinggi adiksi media sosial. "
                     "Pertimbangkan untuk mengurangi screen time dan mengaktifkan screen time limits.")
        else:
            st.success("✅ **Baik!** Profil ini menunjukkan pola penggunaan yang sehat.")

# ══════════════════════════════════════════════════════════════
# PAGE: EVALUASI MODEL
# ══════════════════════════════════════════════════════════════
elif page == "📈 Evaluasi Model":
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

    st.title("📈 Evaluasi Model")

    if not models_ready:
        st.warning("⚠️ Model belum tersedia. Jalankan notebook training terlebih dahulu.")
        st.stop()

    _, scaler, _, feature_names, models_dict = load_model_artifacts()
    test_df = pd.read_csv(os.path.join(ROOT, "data", "processed", "test.csv"))
    X_test = test_df.drop(columns=["addiction_label"]).values
    y_test = test_df["addiction_label"].values

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    CLASS_NAMES = ["Low Addiction", "High Addiction"]

    results = []
    for name, model in models_dict.items():
        yp = model.predict(X_test)
        yprob = model.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy":  round(accuracy_score(y_test, yp), 4),
            "Precision": round(precision_score(y_test, yp, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, yp, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, yp, zero_division=0), 4),
            "ROC-AUC":   round(roc_auc_score(y_test, yprob), 4),
        })
    results_df = pd.DataFrame(results)

    st.subheader("📋 Tabel Perbandingan Model")
    st.dataframe(
        results_df.style.highlight_max(subset=["Accuracy","Precision","Recall","F1-Score","ROC-AUC"],
                                       color="#C8E6C9"),
        use_container_width=True
    )

    st.subheader("📉 ROC Curves")
    fig = go.Figure()
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    for (name, model), color in zip(models_dict.items(), colors):
        yprob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, yprob)
        auc = roc_auc_score(y_test, yprob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.3f})",
                                 line=dict(color=color, width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random", line=dict(color="gray", dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      title="ROC Curves — Perbandingan Model", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔲 Confusion Matrices")
    cols = st.columns(3)
    for col, (name, model) in zip(cols, models_dict.items()):
        yp = model.predict(X_test)
        cm = confusion_matrix(y_test, yp)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           x=CLASS_NAMES, y=CLASS_NAMES,
                           title=name, labels=dict(x="Predicted", y="Actual"))
        col.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("📊 Perbandingan Metrik")
    fig_bar = go.Figure()
    metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colors_bar = ["#2196F3", "#4CAF50", "#FF5722"]
    for (_, row), color in zip(results_df.iterrows(), colors_bar):
        fig_bar.add_trace(go.Bar(name=row["Model"], x=metrics_list,
                                 y=[row[m] for m in metrics_list],
                                 marker_color=color))
    fig_bar.update_layout(barmode="group", yaxis_range=[0, 1.1],
                          title="Perbandingan Semua Metrik", height=450)
    st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: INTERPRETASI
# ══════════════════════════════════════════════════════════════
elif page == "💡 Interpretasi":
    import plotly.express as px

    st.title("💡 Interpretasi Model & Business Insights")

    if not models_ready:
        st.warning("⚠️ Model belum tersedia.")
        st.stop()

    best_model, scaler, _, feature_names, _ = load_model_artifacts()
    test_df = pd.read_csv(os.path.join(ROOT, "data", "processed", "test.csv"))
    _ = test_df["addiction_label"].values

    st.subheader("🔑 Feature Importance (XGBoost)")
    importances = best_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})\
                .sort_values("Importance", ascending=False).head(20)
    fig = px.bar(feat_df.sort_values("Importance"), x="Importance", y="Feature",
                 orientation="h", color="Importance", color_continuous_scale="RdYlGn_r",
                 title="Top 20 Feature Importances")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 SHAP Analysis")
    shap_path = os.path.join(ROOT, "reports", "shap_summary.png")
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.info("Jalankan notebook 03_interpretation.py untuk menghasilkan SHAP plots.")

    st.subheader("💼 Business Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.error("🔴 **Faktor Risiko Utama**\n\n"
                 "- Screen time > 3 jam/hari\n"
                 "- Konsumsi video berlebihan\n"
                 "- Tidur < 6 jam/malam\n"
                 "- Dampak mental health negatif\n"
                 "- Penggunaan 6+ platform sekaligus")
    with col2:
        st.success("🟢 **Rekomendasi Intervensi**\n\n"
                   "- Aktifkan screen time limits\n"
                   "- Batasi notifikasi harian\n"
                   "- Jaga kualitas tidur ≥ 7 jam\n"
                   "- Diversifikasi aktivitas offline\n"
                   "- Gunakan privacy settings")

    st.markdown("---")
    st.subheader("🎯 Justifikasi Model Terbaik: XGBoost")
    st.markdown("""
**XGBoost dipilih sebagai model terbaik** berdasarkan:

1. **ROC-AUC Tertinggi** — Kemampuan diskriminasi terbaik antar kelas
2. **F1-Score Seimbang** — Performa baik pada kelas mayoritas maupun minoritas
3. **Penanganan Imbalance** — `scale_pos_weight` menyeimbangkan pengaruh kelas
4. **Feature Importance** — Gain-based importance lebih informatif untuk interpretasi
5. **Robustness** — Menangkap interaksi non-linear antar fitur yang tidak bisa
   ditangkap Logistic Regression
6. **Hyperparameter Tuning** — RandomizedSearchCV dengan 20 iterasi mengoptimalkan
   kombinasi depth, learning rate, subsample, dan colsample_bytree
    """)

# ══════════════════════════════════════════════════════════════
# PAGE: DOKUMENTASI
# ══════════════════════════════════════════════════════════════
elif page == "📚 Dokumentasi":
    st.title("📚 Dokumentasi Proyek")

    st.subheader("📁 Dataset")
    st.markdown("""
**Social Media User Behavior Dataset**
- **Sumber:** Kaggle / Dataset publik
- **Ukuran:** 25.000 baris × 45 kolom
- **Target:** `addiction_level_1_to_10` → binarisasi menjadi `addiction_label`
- **Fitur:** Demografis, perilaku penggunaan, aktivitas, dampak psikologis

**Platform Statistics 2026:**
- Data referensi 9 platform media sosial utama
    """)
    df = load_raw_data()
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.subheader("⚙️ Metodologi")
    st.markdown("""
```
Raw Data (25,000 rows)
    │
    ▼
Preprocessing
    ├── Drop: user_id, account_created_date
    ├── Encode boolean → int
    ├── LabelEncode kategorikal (14 kolom)
    ├── StandardScaler pada fitur numerik
    └── Train/Val/Test split (70/10/20, stratified)
    │
    ▼
Modeling & Tuning (RandomizedSearchCV, 5-fold CV)
    ├── Logistic Regression (baseline)
    ├── Random Forest
    └── XGBoost ← Best Model
    │
    ▼
Evaluation
    ├── Accuracy, Precision, Recall, F1, ROC-AUC
    ├── Confusion Matrix
    └── ROC Curves
    │
    ▼
Interpretation
    ├── Feature Importance (Gain-based)
    └── SHAP TreeExplainer
    │
    ▼
Deployment (Streamlit)
```
    """)

    st.subheader("📦 Dependencies")
    with open(os.path.join(ROOT, "requirements.txt")) as f:
        st.code(f.read(), language="text")
