import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from churn_pipeline.config import ARTIFACT_VERSION, FEATURE_COLUMNS, TARGET_MAPPING
from churn_pipeline.data import prepare_features
from churn_pipeline.models import build_model_pipeline
from churn_pipeline.predict import score_customers, load_artifacts, runtime_versions
from churn_pipeline.train import save_artifacts


def fitted_model(customers):
    X = prepare_features(customers)
    y = customers["Churn"].map(TARGET_MAPPING)
    return build_model_pipeline("logistic").fit(X, y)


def test_score_customers_contract(raw_customers):
    pipeline = fitted_model(raw_customers)
    data = raw_customers.iloc[::-1].drop(columns="Churn")
    before = data.copy(deep=True)
    result = score_customers(data, pipeline, .5)
    assert result.columns.tolist() == ["customerID", "churn_probability", "predicted_class", "risk_tier"]
    assert result.customerID.tolist() == data.customerID.tolist()
    assert result.churn_probability.between(0, 1).all()
    assert (result.predicted_class == (result.churn_probability >= .5).astype(int)).all()
    assert (result.risk_tier == np.where(result.predicted_class == 1, "High", "Standard")).all()
    pd.testing.assert_frame_equal(data, before)
    extra = data.assign(Churn="Yes", ChurnFlag=1)
    pd.testing.assert_frame_equal(result, score_customers(extra, pipeline, .5))
    unseen = data.copy()
    unseen["InternetService"] = "unseen service"
    assert len(score_customers(unseen, pipeline, .5)) == len(unseen)
    with pytest.raises(ValueError):
        score_customers(data.drop(columns="tenure"), pipeline, .5)


def test_predict_cli(raw_customers, tmp_path):
    pipeline = fitted_model(raw_customers)
    metadata = {
        "artifact_version": ARTIFACT_VERSION, "feature_columns": FEATURE_COLUMNS,
        "target_mapping": TARGET_MAPPING, "threshold": .5, "versions": runtime_versions(),
    }
    model, meta = save_artifacts(pipeline, metadata, tmp_path)
    source, output = tmp_path / "customers.csv", tmp_path / "predictions.csv"
    raw_customers.to_csv(source, index=False)
    command = [sys.executable, "-m", "churn_pipeline.predict", "--input", str(source),
               "--output", str(output), "--model", str(model), "--metadata", str(meta)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "Churn" in run.stdout
    result = pd.read_csv(output, dtype={"customerID": "string"})
    assert result.customerID.tolist() == raw_customers.customerID.tolist()
    np.testing.assert_allclose(result.churn_probability, score_customers(raw_customers, pipeline, .5).churn_probability)
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert subprocess.run(command + ["--overwrite"], capture_output=True).returncode == 0
    input_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    same_path = command.copy()
    same_path[same_path.index("--output") + 1] = str(source)
    assert subprocess.run(same_path + ["--overwrite"], capture_output=True).returncode != 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == input_hash
    invalid = raw_customers.drop(columns="tenure")
    invalid.to_csv(source, index=False)
    assert subprocess.run(command + ["--overwrite"], capture_output=True).returncode != 0
    record = json.loads(meta.read_text())
    record["model_sha256"] = "wrong"
    meta.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="哈希"):
        load_artifacts(model, meta)
