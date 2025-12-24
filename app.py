import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Judul aplikasi
st.title("🚢 Titanic Survival Prediction App")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df = df[["Survived", "Pclass", "Sex", "Age", "Fare", "SibSp"]].dropna()
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
    return df

df = load_data()

# Fitur dan target
X = df[["Pclass", "Sex", "Age", "Fare", "SibSp"]]
y = df["Survived"]

# Train model langsung
model = RandomForestClassifier()
model.fit(X, y)

# Sidebar input
st.sidebar.header("🧾 Data Penumpang")

pclass = st.sidebar.selectbox("Kelas Tiket", [1, 2, 3])

desc_kelas = {
    1: "1 (First Class) — Penumpang kelas atas dengan fasilitas terbaik.",
    2: "2 (Second Class) — Penumpang kelas menengah.",
    3: "3 (Third Class) — Penumpang kelas ekonomi / bawah."
}
st.sidebar.markdown(f"**Keterangan Kelas Tiket:**  \n{desc_kelas[pclass]}")

sex = st.sidebar.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
age = st.sidebar.slider("Usia", 0, 80, 30)
fare = st.sidebar.slider("Harga Tiket (Fare)", 0.0, 600.0, 50.0)
sibsp = st.sidebar.slider("Jumlah Saudara/Suami/Istri", 0, 5, 0)

# Encode jenis kelamin
sex_encoded = 1 if sex == "Laki-laki" else 0

# Siapkan fitur input untuk prediksi
features = np.array([[pclass, sex_encoded, age, fare, sibsp]])

# Prediksi
prediction = model.predict(features)[0]
proba = model.predict_proba(features)[0][1]

# Tampilkan hasil prediksi
st.subheader("🎯 Hasil Prediksi")
if prediction == 1:
    st.success("Penumpang kemungkinan **🟢 Selamat**")
else:
    st.error("Penumpang kemungkinan **🔴 Tidak Selamat**")

st.metric("Probabilitas Selamat", f"{proba:.2%}")

# Visualisasi distribusi usia penumpang
st.subheader("📊 Distribusi Umur Penumpang Titanic")

df_vis = df.copy()
df_vis["SurvivedLabel"] = df_vis["Survived"].map({0: "Tidak Selamat", 1: "Selamat"})
df_vis["SexLabel"] = df_vis["Sex"].map({1: "Laki-laki", 0: "Perempuan"})

fig_age = px.histogram(
    df_vis,
    x="Age",
    color="SurvivedLabel",
    color_discrete_map={"Selamat": "green", "Tidak Selamat": "red"},
    nbins=30,
    title="Distribusi Umur vs Keselamatan",
    labels={"Age": "Umur", "count": "Jumlah Penumpang"}
)
st.plotly_chart(fig_age)

# Visualisasi survival rate berdasarkan kelas tiket dan jenis kelamin
st.subheader("📊 Survival Rate Berdasarkan Kelas Tiket dan Jenis Kelamin")

fig_survival = px.histogram(
    df_vis,
    x="Pclass",
    color="SurvivedLabel",
    barmode="group",
    facet_col="SexLabel",
    category_orders={"Pclass": [1, 2, 3]},
    color_discrete_map={"Selamat": "green", "Tidak Selamat": "red"},
    labels={"Pclass": "Kelas Tiket", "count": "Jumlah Penumpang"},
    title="Survival Rate Berdasarkan Kelas Tiket dan Jenis Kelamin"
)
st.plotly_chart(fig_survival)
