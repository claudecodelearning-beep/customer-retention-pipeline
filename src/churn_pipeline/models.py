from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from .preprocessing import build_preprocessor
from .config import BASE_XGB_PARAMS


def build_model_pipeline(model_name, *, xgb_params=None, random_state=42):
    if model_name == "dummy":
        model = DummyClassifier(strategy="prior")
    elif model_name == "logistic":
        model = LogisticRegression(max_iter=2000, solver="lbfgs")
    elif model_name == "xgboost":
        params = {**BASE_XGB_PARAMS, **(xgb_params or {})}
        params.update(objective="binary:logistic", eval_metric="logloss",
                      tree_method="hist", random_state=random_state, n_jobs=1, verbosity=0)
        model = XGBClassifier(**params)
    else:
        raise ValueError(f"未知模型名称: {model_name}")
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=model_name != "xgboost")),
        ("model", model),
    ])
