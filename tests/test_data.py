import numpy as np
import pandas as pd
import pytest
from churn_pipeline.config import FEATURE_COLUMNS
from churn_pipeline.data import load_customers, prepare_features, split_customers, validate_customers


def test_prepare_features_preserves_input(raw_customers):
    raw_customers.loc[0, "TotalCharges"] = "   "
    raw_customers.loc[0, "SeniorCitizen"] = np.nan
    raw_customers["ChurnFlag"] = 1
    before = raw_customers.copy(deep=True)
    result = prepare_features(raw_customers)
    assert result.columns.tolist() == FEATURE_COLUMNS
    assert pd.isna(result.loc[0, "TotalCharges"])
    assert pd.isna(result.loc[0, "SeniorCitizen"])
    assert result.loc[1, "SeniorCitizen"] == "0"
    pd.testing.assert_frame_equal(raw_customers, before)


@pytest.mark.parametrize("problem", ["missing", "duplicate_id", "blank_id", "target", "text", "inf", "columns"])
def test_invalid_customer_data(raw_customers, problem):
    data = raw_customers.copy()
    if problem == "missing":
        data = data.drop(columns="tenure")
    elif problem == "duplicate_id":
        data.loc[1, "customerID"] = data.loc[0, "customerID"]
    elif problem == "blank_id":
        data.loc[0, "customerID"] = "  "
    elif problem == "target":
        data.loc[0, "Churn"] = "unknown"
    elif problem == "text":
        data.loc[0, "TotalCharges"] = "oops"
    elif problem == "inf":
        data.loc[0, "MonthlyCharges"] = np.inf
    else:
        data.columns = ["customerID"] * len(data.columns)
    with pytest.raises(ValueError):
        validate_customers(data, require_target=True)


def test_split_reproducible_and_aligned(raw_customers):
    first = split_customers(raw_customers)
    second = split_customers(raw_customers)
    for a, b in zip(first, second):
        assert a.equals(b)
    X_train, X_test, y_train, y_test, ids_train, ids_test = first
    assert X_train.index.equals(y_train.index) and X_train.index.equals(ids_train.index)
    assert X_test.index.equals(y_test.index) and X_test.index.equals(ids_test.index)
    assert set(ids_train).isdisjoint(ids_test)


def test_csv_contract(raw_customers, tmp_path):
    path = tmp_path / "input.csv"
    raw_customers.to_csv(path, index=False)
    assert load_customers(path).loc[0, "customerID"] == "0000"
    path.write_text("customerID,customerID\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_customers(path)
    with pytest.raises(FileNotFoundError):
        load_customers(tmp_path / "missing.csv")
