import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------------
# Page config
# -----------------------------------------------------------
st.set_page_config(page_title="DSS Iklim Papua Kabupaten Biak Numfor", layout="wide")

# -----------------------------------------------------------
# CUSTOM BACKGROUND (SOFT SILVER GRAY)
# -----------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f2f2f2;   /* light soft gray */
    }
    .stApp {
        background-color: #e6e6e6;   /* silver gray */
    }

    /* Optional: styling boxes/cards */
    .css-1d391kg, .css-1kyxreq, .css-1r6slb0 {
        background-color: white !important;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Header
# -----------------------------------------------------------
st.title("🌦️ Decision Support System Iklim - Papua Kabupaten Biak Numfor")
st.markdown("Mendukung literasi iklim dan berpikir komputasi calon guru fisika melalui analisis data cuaca harian.")

# -----------------------------------------------------------
# Helper Functions DSS
# -----------------------------------------------------------
def klasifikasi_cuaca(ch, matahari):
    try:
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
        return "Ya" if ch > 50 else "Tidak"
    except:
        return "N/A"

# -----------------------------------------------------------
# Load data otomatis tanpa upload
# -----------------------------------------------------------
DATA_PATH = "Papua Biaknumfor.xlsx"
st.sidebar.success("📂 Data dimuat otomatis dari file lokal: Papua Biaknumfor.xlsx")

@st.cache_data
def process_data(_):  # dummy param
    df = pd.read_excel(DATA_PATH, sheet_name="Data Harian - Table")

    # pastikan tidak ada duplikasi kolom
    df = df.loc[:, ~df.columns.duplicated()]

    # rename jika ada kolom kecepatan angin lama
    if "kecepatan_angin" in df.columns:
        df = df.rename(columns={"kecepatan_angin": "FF_X"})

    # parse tanggal
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True, errors='coerce')

    # lengkapi kolom yang mungkin hilang
    expected_cols = ['Tn','Tx','Tavg','kelembaban','curah_hujan','matahari','FF_X','DDD_X']
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    # tambah fitur tanggal
    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan"] = df["Tanggal"].dt.month
    df["Hari"] = df["Tanggal"].dt.day

    # DSS kategori
    df["Prediksi Cuaca"] = df.apply(lambda r: klasifikasi_cuaca(r["curah_hujan"], r["matahari"]), axis=1)
    df["Risiko Kekeringan"] = df.apply(lambda r: risiko_kekeringan(r["curah_hujan"], r["matahari"]), axis=1)
    df["Hujan Ekstrem"] = df["curah_hujan"].apply(hujan_ekstrem)

    return df

df = process_data("dummy")

# -----------------------------------------------------------
# Tampilkan Data
# -----------------------------------------------------------
st.subheader("📋 Data Iklim Harian")
st.dataframe(df, use_container_width=True)

# -----------------------------------------------------------
# Grafik Tren CH dan Suhu
# -----------------------------------------------------------
st.subheader("📈 Tren Curah Hujan & Suhu")

col1, col2 = st.columns(2)

with col1:
    fig_ch = px.line(df, x="Tanggal", y="curah_hujan", title="Tren Curah Hujan (mm)")
    st.plotly_chart(fig_ch, use_container_width=True)

with col2:
    fig_temp = px.line(df, x="Tanggal", y="Tavg", title="Tren Suhu Rata-Rata (°C)")
    st.plotly_chart(fig_temp, use_container_width=True)

# -----------------------------------------------------------
# Prediksi CH & Suhu 10–50 tahun
# -----------------------------------------------------------
st.subheader("🤖 Prediksi Iklim Menggunakan Machine Learning (RF Regressor)")

# pilih fitur
features = ["Tahun","Bulan","Hari","Tn","Tx","Tavg","kelembaban","FF_X","DDD_X"]
target_ch = "curah_hujan"
target_temp = "Tavg"

df_model = df.dropna(subset=features + [target_ch, target_temp])

X = df_model[features]
y_ch = df_model[target_ch]
y_temp = df_model[target_temp]

X_train, X_test, y_train_ch, y_test_ch = train_test_split(X, y_ch, test_size=0.2, random_state=42)
_, _, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)

# train model
model_ch = RandomForestRegressor(n_estimators=200, random_state=42)
model_temp = RandomForestRegressor(n_estimators=200, random_state=42)

model_ch.fit(X_train, y_train_ch)
model_temp.fit(X_train, y_train_temp)

# buat prediksi tahun 2030–2070
future_years = st.slider("Pilih rentang tahun prediksi:", 2030, 2070, (2030, 2050))
years = list(range(future_years[0], future_years[1] + 1))

future_df = pd.DataFrame({
    "Tahun": years,
    "Bulan": [1]*len(years),
    "Hari": [1]*len(years),
    "Tn": [df["Tn"].mean()]*len(years),
    "Tx": [df["Tx"].mean()]*len(years),
    "Tavg": [df["Tavg"].mean()]*len(years),
    "kelembaban": [df["kelembaban"].mean()]*len(years),
    "FF_X": [df["FF_X"].mean()]*len(years),
    "DDD_X": [df["DDD_X"].mean()]*len(years),
})

future_df["Prediksi CH"] = model_ch.predict(future_df[features])
future_df["Prediksi Suhu"] = model_temp.predict(future_df[features])

st.write("### 📊 Hasil Prediksi")
st.dataframe(future_df, use_container_width=True)

# grafik prediksi
fig_future_ch = px.line(future_df, x="Tahun", y="Prediksi CH", title="Prediksi Curah Hujan (mm)")
st.plotly_chart(fig_future_ch, use_container_width=True)

fig_future_temp = px.line(future_df, x="Tahun", y="Prediksi Suhu", title="Prediksi Suhu (°C)")
st.plotly_chart(fig_future_temp, use_container_width=True)
