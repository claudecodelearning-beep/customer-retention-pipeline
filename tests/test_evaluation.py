import numpy as np
import pytest
from churn_pipeline.data import prepare_features
from churn_pipeline.models import build_model_pipeline
from churn_pipeline.evaluation import metrics_at_threshold, select_capacity_threshold, make_cv_splits, generate_oof_probability


def test_threshold_equality():
    result = metrics_at_threshold([0, 1, 1], [0.1, 0.5, 0.9], 0.5)
    assert [result[k] for k in ["tn", "fp", "fn", "tp"]] == [1, 0, 0, 2]


@pytest.mark.parametrize("y,p,t", [([0], [np.nan], .5), ([0], [1.2], .5), ([0], [.1, .2], .5), ([0], [.1], np.inf)])
def test_invalid_metrics(y, p, t):
    with pytest.raises(ValueError):
        metrics_at_threshold(y, p, t)


def test_capacity_ties():
    assert select_capacity_threshold([.9, .8, .8, .1], capacity_share=.5) == .9
    with pytest.raises(ValueError):
        select_capacity_threshold([.8] * 4, capacity_share=.5)


def test_oof_coverage(raw_customers):
    X = prepare_features(raw_customers)
    y = raw_customers["Churn"].map({"No": 0, "Yes": 1})
    folds = make_cv_splits(X, y, n_splits=3)
    probability = generate_oof_probability(build_model_pipeline("logistic"), X, y, folds)
    assert probability.shape == (len(X),)
    assert np.isfinite(probability).all()
    with pytest.raises(ValueError, match="一次"):
        generate_oof_probability(build_model_pipeline("logistic"), X, y, folds[:-1])
