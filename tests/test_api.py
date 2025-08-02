"""Tests for the FastAPI prediction endpoint."""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add project root to Python path and configure token before import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ["API_TOKEN"] = "testtoken"

from api.main import app  # noqa: E402

client = TestClient(app)


sample_payload = {
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


def test_predict_authorized():
    response = client.post(
        "/predict",
        json=sample_payload,
        headers={"Authorization": "Bearer testtoken"},
    )
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_requires_auth():
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 403

