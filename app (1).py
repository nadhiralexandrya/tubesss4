
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Risiko Stroke", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load('model_stroke.pkl')

model = load_model()

st.title("🧠 Aplikasi Prediksi Risiko Stroke")
st.markdown("Masukkan data pasien di bawah ini untuk memprediksi risiko stroke.")
st.markdown("---")

st.subheader("📋 Data Pasien")
col1, col2 = st.columns(2)

with col1:
    gender            = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age               = st.number_input("Usia (tahun)", min_value=1, max_value=100, value=45)
    hypertension      = st.selectbox("Hipertensi", ["Tidak", "Ya"])
    heart_disease     = st.selectbox("Penyakit Jantung", ["Tidak", "Ya"])
    ever_married      = st.selectbox("Status Pernikahan", ["Yes", "No"])

with col2:
    residence_type    = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Rata-rata Kadar Glukosa", min_value=50.0, max_value=300.0, value=100.0)
    bmi               = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    work_type         = st.selectbox("Jenis Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    smoking_status    = st.selectbox("Status Merokok", ["never smoked", "formerly smoked", "smokes", "Unknown"])

st.markdown("---")

def preprocess_input():
    return pd.DataFrame([{
        'gender'                         : 1 if gender == "Male" else 0,
        'age'                            : age,
        'hypertension'                   : 1 if hypertension == "Ya" else 0,
        'heart_disease'                  : 1 if heart_disease == "Ya" else 0,
        'ever_married'                   : 1 if ever_married == "Yes" else 0,
        'Residence_type'                 : 1 if residence_type == "Urban" else 0,
        'avg_glucose_level'              : avg_glucose_level,
        'bmi'                            : bmi,
        'work_type_Govt_job'             : 1 if work_type == "Govt_job" else 0,
        'work_type_Never_worked'         : 1 if work_type == "Never_worked" else 0,
        'work_type_Private'              : 1 if work_type == "Private" else 0,
        'work_type_Self-employed'        : 1 if work_type == "Self-employed" else 0,
        'work_type_children'             : 1 if work_type == "children" else 0,
        'smoking_status_Unknown'         : 1 if smoking_status == "Unknown" else 0,
        'smoking_status_formerly smoked' : 1 if smoking_status == "formerly smoked" else 0,
        'smoking_status_never smoked'    : 1 if smoking_status == "never smoked" else 0,
        'smoking_status_smokes'          : 1 if smoking_status == "smokes" else 0,
    }])

if st.button("🔍 Prediksi Sekarang", use_container_width=True, type="primary"):
    input_df        = preprocess_input()
    prediction      = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0]

    prob_tidak = prediction_prob[0] * 100
    prob_stroke = prediction_prob[1] * 100

    st.subheader("📊 Hasil Prediksi")
    if prediction == 1:
        st.error(f"⚠️ BERISIKO STROKE ({prob_stroke:.1f}% probabilitas)")
        st.warning("Segera konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.")
    else:
        st.success(f"✅ TIDAK BERISIKO STROKE ({prob_tidak:.1f}% probabilitas)")
        st.info("Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Tidak Stroke", f"{prob_tidak:.1f}%")
        st.progress(prob_tidak / 100)
    with col_b:
        st.metric("Stroke", f"{prob_stroke:.1f}%")
        st.progress(prob_stroke / 100)

    st.markdown("---")
    st.subheader("📝 Ringkasan Data Input")
    ringkasan = {
        "Jenis Kelamin": gender, "Usia": f"{age} tahun",
        "Hipertensi": hypertension, "Penyakit Jantung": heart_disease,
        "Status Pernikahan": ever_married, "Tempat Tinggal": residence_type,
        "Kadar Glukosa": avg_glucose_level, "BMI": bmi,
        "Pekerjaan": work_type, "Status Merokok": smoking_status,
    }
    st.table(pd.DataFrame(ringkasan.items(), columns=["Fitur", "Nilai"]))

st.markdown("---")
st.caption("⚕️ Disclaimer: Aplikasi ini hanya untuk keperluan akademik.")
