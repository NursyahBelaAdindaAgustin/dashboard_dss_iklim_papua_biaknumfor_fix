# app.py — DSS Iklim Papua Kab. Biak Numfor (Final with CARD NAVIGATION)
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------
# Config
# ----------------------
st.set_page_config(page_title="DSS Iklim Papua Kabupaten Biak Numfor", layout="wide")
st.title("🌦️ Decision Support System Iklim - Papua Kabupaten Biak Numfor")
st.markdown("Mendukung literasi iklim dan berpikir komputasi calon guru fisika melalui analisis data cuaca harian.")

DATA_PATH = "Papua Biaknumfor.xlsx"  # pastikan file ini ada di folder proyek

# ----------------------
# CARD NAVIGATION (NO BACKGROUND EDIT)
# ----------------------
st.markdown("""
<style>
.card-container {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}

.card {
    flex: 1;
    min-width: 240px;
    background: white;
    padding: 16px;
    border-radius: 12px;
    text-align: left;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    transition: 0.2s ease;
    border: 1px solid #eeeeee;
    text-decoration: none;
    color: inherit;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
    border-color: #cccccc;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 4px;
}

.card-desc {
    font-size: 14px;
    opacity: 0.7;
}
</style>

<div class="card-container">

    <a href="#data-section" class="card">
        <div class="card-title">📄 Data Cuaca</div>
        <div class="card-desc">Tabel data harian wilayah Biak Numfor</div>
    </a>

    <a href="#visual-section" class="card">
        <div class="card-title">📊 Visualisasi</div>
        <div class="card-desc">Grafik trend curah hujan, suhu, kelembaban</div>
    </a>

    <a href="#dss-section" class="card">
        <div class="card-title">🔎 Analisis DSS</div>
        <div class="card-desc">Klasifikasi cuaca, cuaca ekstrem, risiko kekeringan</div>
    </a>

    <a href="#prediksi-section" class="card">
        <div class="card-title">🤖 Prediksi ML</div>
        <div class="card-desc">Random Forest untuk prediksi 10–50 tahun ke depan</div>
    </a>

</div>
""", unsafe_allow_html=True)

# ----------------------
# Helper functions (DSS)
# ----------------------
def klasifikasi_cuaca(ch, matahari):
    try:
        if pd.isna(ch) and pd.isna(matahari):
            return "N/A"
        if ch > 20:
            return "Hujan"
        elif ch > 5:
            return "Berawan"
        elif matahari > 4:
            return "Cerah"
        else:
            return "Berawan"
    except:
        return "N/A"

def risiko_kekeringan(ch, matahari):
    try:
        if pd.isna(ch) and pd.isna(matahari):
            return "N/A"
        if ch < 1 and matahari > 6:
            return "Risiko Tinggi"
        elif ch < 5:
            return "Risiko Sedang"
        else:
            return "Risiko Rendah"
    except:
        return "N/A"

def hujan_ekstrem(ch):
    try:
        if pd.isna(ch):
            return "N/A"
        return "Ya" if ch > 50 else "Tidak"
    except:
        return "N/A"

# ----------------------
# Load & Process Data
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name="Data Harian - Table")
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True, errors="coerce")

    # ganti nama kolom jika perlu
    if "kecepatan_angin" in df.columns and "FF_X" not in df.columns:
        df = df.rename(columns={"kecepatan_angin": "FF_X"})

    # Kolom wajib
    expected_cols = ["Tn","Tx","Tavg","kelembaban","curah_hujan","matahari","FF_X","DDD_X"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    # DSS tambahan
    df["Klasifikasi Cuaca"] = df.apply(lambda r: klasifikasi_cuaca(r["curah_hujan"], r["matahari"]), axis=1)
    df["Hujan Ekstrem"] = df["curah_hujan"].apply(hujan_ekstrem)
    df["Risiko Kekeringan"] = df.apply(lambda r: risiko_kekeringan(r["curah_hujan"], r["matahari"]), axis=1)

    return df

df = load_data()

# ----------------------
# SECTION — DATA
# ----------------------
st.markdown("## 📄 Data Cuaca", unsafe_allow_html=True)
st.markdown('<div id="data-section"></div>', unsafe_allow_html=True)

st.dataframe(df, use_container_width=True)

# ----------------------
# SECTION — VISUALISASI
# ----------------------
st.markdown("## 📊 Visualisasi", unsafe_allow_html=True)
st.markdown('<div id="visual-section"></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_temp = px.line(df, x="Tanggal", y=["Tn","Tx","Tavg"], title="Perubahan Suhu Harian")
    st.plotly_chart(fig_temp, use_container_width=True)

with col2:
    fig_ch = px.line(df, x="Tanggal", y="curah_hujan", title="Trend Curah Hujan Harian")
    st.plotly_chart(fig_ch, use_container_width=True)

# ----------------------
# SECTION — DSS
# ----------------------
st.markdown("## 🔎 Analisis DSS (Decision Support System)", unsafe_allow_html=True)
st.markdown('<div id="dss-section"></div>', unsafe_allow_html=True)

st.write("### Contoh Statistik DSS")
st.write(df[["Klasifikasi Cuaca", "Hujan Ekstrem", "Risiko Kekeringan"]].value_counts())

# ----------------------
# SECTION — PREDIKSI IKLIM (ML)
# ----------------------
st.markdown("## 🤖 Prediksi Iklim Menggunakan Random Forest", unsafe_allow_html=True)
st.markdown('<div id="prediksi-section"></div>', unsafe_allow_html=True)

feature_cols = ["Tn","Tx","Tavg","kelembaban","matahari","FF_X"]
df_ml = df.dropna(subset=feature_cols + ["curah_hujan"])

X = df_ml[feature_cols]
y = df_ml["curah_hujan"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("**R² Score:**", r2_score(y_test, y_pred))
st.write("**RMSE:**", np.sqrt(mean_squared_error(y_test, y_pred)))

# Prediksi masa depan
tahun_ke = st.slider("Prediksi berapa tahun ke depan?", 10, 50, 20)
pred_future = model.predict([X.mean().tolist()])[0]

st.success(f"📈 **Prediksi Curah Hujan Rata-rata {tahun_ke} tahun ke depan: {pred_future:.2f} mm/hari**")
