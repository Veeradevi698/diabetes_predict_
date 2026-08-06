import streamlit as st
import numpy as np
import pickle

# Load saved model and scaler
with open("finalized_model.sav", "rb") as f:
    model = pickle.load(f)

with open("scaler_model.sav", "rb") as f:
    scaler = pickle.load(f)

# Title
st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details below to predict diabetes outcome:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction button
if st.button("Predict"):
    # Prepare input
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]).reshape(1, -1)

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    # Output
    if prediction[0] == 1:
        st.error(f"⚠️ Diabetes Positive (Probability: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Diabetes Negative (Probability: {(1-probability)*100:.2f}%)")
