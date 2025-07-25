import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5em 1em;
            font-size: 16px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("📉 Customer Churn Prediction")
st.write("Enter customer details to predict if they are likely to churn.")

# Input form
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    age = col1.slider("Age", 18, 100, 30)
    gender = col2.selectbox("Gender", ["Male", "Female"])
    tenure = col1.slider("Tenure (in months)", 0, 72, 12)
    monthly_charges = col2.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Gender encoding
    gender_encoded = 1 if gender == "Female" else 0  # 1 for Female, 0 for Male

    # Create input DataFrame with feature names
    user_data = pd.DataFrame([[age, gender_encoded, tenure, monthly_charges]],
                             columns=['Age', 'Gender', 'Tenure', 'MonthlyCharges'])

    # Scale the input
    scaled_data = scaler.transform(user_data)

    # Predict
    prediction = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    # Output
    if prediction == 1:
        st.error(f"⚠️ Likely to Churn — Probability: {prob:.2%}")
    else:
        st.success(f"✅ Not Likely to Churn — Probability: {prob:.2%}")
