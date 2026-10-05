import pandas as pd
import pytest


@pytest.fixture
def raw_customers():
    """人工数据，不读取真实客户 CSV；每次测试得到一个新对象。"""
    row = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 10, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
        "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0, "TotalCharges": "500",
    }
    records = []
    for i in range(12):
        records.append({
            **row, "customerID": f"{i:04d}", "tenure": i + 1,
            "MonthlyCharges": 40.0 + i, "TotalCharges": str((i + 1) * (40 + i)),
            "Churn": "Yes" if i % 2 else "No",
        })
    return pd.DataFrame(records)
