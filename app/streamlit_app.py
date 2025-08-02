"""Streamlit demo application for the credit score API."""

import os
from typing import Dict

import requests
import streamlit as st

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
API_TOKEN = os.getenv("API_TOKEN", "secret-token")

st.title("QuantumFinance Credit Score")
st.write("Preencha os dados do cliente e obtenha a previsão do score de crédito.")

default_payload: Dict[str, object] = {
    "Age": 35,
    "Annual_Income": 65000,
    "Num_Bank_Accounts": 2,
    "Num_Credit_Card": 1,
    "Interest_Rate": 10,
    "Num_of_Loan": 3,
    "Delay_from_due_date": 2,
    "Num_of_Delayed_Payment": 1,
    "Changed_Credit_Limit": 5,
    "Num_Credit_Inquiries": 2,
    "Credit_Mix": "Good",
    "Outstanding_Debt": 1500,
    "Payment_Behaviour": "High_spent_Large_value_payments",
    "Payment_of_Min_Amount": "Yes",
    "Total_EMI_per_month": 300,
    "Type_of_Loan": "Auto Loan",
    "Amount_invested_monthly": 500,
    "Credit_History_Age": "3 Years and 5 Months",
    "Monthly_Balance": 1200,
    "Occupation": "Engineer",
    "Monthly_Inhand_Salary": 5000,
    "Credit_Utilization_Ratio": 0.35,
}

numeric_fields = {
    "Age",
    "Annual_Income",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Monthly_Inhand_Salary",
    "Credit_Utilization_Ratio",
}

payload: Dict[str, object] = {}
for field, default in default_payload.items():
    if field in numeric_fields:
        payload[field] = st.number_input(field, value=float(default))
    else:
        payload[field] = st.text_input(field, value=str(default))

if st.button("Predict"):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    with st.spinner("Consultando API..."):
        response = requests.post(API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        st.success(f"Predição: {response.json()['prediction']}")
    else:
        st.error(f"Erro: {response.text}")

