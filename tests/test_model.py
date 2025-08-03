import joblib
import pandas as pd
import os

def load_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.abspath(os.path.join(current_dir, "..", "models", "model.pkl"))
    return joblib.load(model_path)

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
        "Credit_Utilization_Ratio": 0.35
    }])

    pred = model.predict(sample)
    assert len(pred) == 1

def test_prediction_for_risky_client():
    model = load_model()
    sample = pd.DataFrame([{
        "Annual_Income": 10000.0,
        "Monthly_Inhand_Salary": 500.0,
        "Num_Bank_Accounts": 8,
        "Num_Credit_Card": 6,
        "Interest_Rate": 28.0,
        "Num_of_Loan": 5,
        "Delay_from_due_date": 30,
        "Num_of_Delayed_Payment": 12,
        "Changed_Credit_Limit": 50000.0,
        "Num_Credit_Inquiries": 0,
        "Credit_Utilization_Ratio": 0.0,
        "Outstanding_Debt": 0.0,
        "Monthly_Balance": 0.0,
        "Age": 18,
        "Total_EMI_per_month": 0.0,
        "Type_of_Loan": "Personal Loan",
        "Payment_Behaviour": "Low_spent_Large_value_payments",
        "Amount_invested_monthly": 0.0,
        "Credit_Mix": "Bad",
        "Payment_of_Min_Amount": "No",
        "Credit_History_Age": "1 Year and 2 Months",
        "Occupation": "Unemployed"
    }])

    pred = model.predict(sample)
    assert len(pred) == 1
    print(f"🧪 Previsão do modelo para perfil de risco: {pred[0]}")
