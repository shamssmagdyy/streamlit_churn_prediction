import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("gym_churn_model.pkl")

st.title("Gym Customer Churn Prediction 💪")

st.write("Enter customer details:")

# User Inputs
avg_total = st.number_input("Avg Class Frequency Total", min_value=0.0)
avg_month = st.number_input("Avg Class Frequency Current Month", min_value=0.0)
lifetime = st.number_input("Lifetime (months)", min_value=0)
month_to_end = st.number_input("Months to End Contract", min_value=0)
contract_period = st.number_input("Contract Period", min_value=0)

# Create DataFrame
input_data = pd.DataFrame({
    "Avg_class_frequency_total": [avg_total],
    "Avg_class_frequency_current_month": [avg_month],
    "Lifetime": [lifetime],
    "Month_to_end_contract": [month_to_end],
    "Contract_period": [contract_period]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Customer will STAY\nProbability of churn: {probability:.2%}")