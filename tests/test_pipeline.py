import joblib
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from churn_pipeline.data import prepare_features
from churn_pipeline.preprocessing import build_preprocessor
from churn_pipeline.models import build_model_pipeline


def test_preprocessor_is_unfitted():
    first = build_preprocessor(scale_numeric=True)
    assert first is not build_preprocessor(scale_numeric=True)
    with pytest.raises(NotFittedError):
        check_is_fitted(first)
    assert "scaler" in first.transformers[0][1].named_steps
    assert "scaler" not in build_preprocessor(scale_numeric=False).transformers[0][1].named_steps


def test_preprocessing_uses_training_only(raw_customers):
    X = prepare_features(raw_customers)
    train, valid = X.iloc[:8].copy(), X.iloc[8:].copy()
    valid["tenure"] = 10000
    valid["InternetService"] = "new service"
    prep = build_preprocessor(scale_numeric=True).fit(train)
    imputer = prep.named_transformers_["numeric"].named_steps["imputer"]
    stats = imputer.statistics_.copy()
    assert stats[0] == train["tenure"].median()
    assert prep.transform(valid).shape[1] == prep.transform(train).shape[1]
    np.testing.assert_array_equal(stats, imputer.statistics_)


def test_model_roundtrip(raw_customers, tmp_path):
    X = prepare_features(raw_customers)
    y = raw_customers["Churn"].map({"No": 0, "Yes": 1})
    pipeline = build_model_pipeline("xgboost", xgb_params={"n_estimators": 3, "max_depth": 2}).fit(X, y)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    np.testing.assert_allclose(pipeline.predict_proba(X), joblib.load(path).predict_proba(X))
