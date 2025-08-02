# tests/test_model.py

import joblib
import pandas as pd

def load_model():
    return joblib.load("models/model.pkl")  

def test_model_loads():
    model = load_model()
    assert model is not None

def test_model_prediction_shape():
    model = load_model()
    sample = pd.DataFrame([{
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
        "Credit_Utilization_Ratio": 0.35  # ← FALTAVA ESSA!
    }])

    pred = model.predict(sample)
    assert len(pred) == 1
