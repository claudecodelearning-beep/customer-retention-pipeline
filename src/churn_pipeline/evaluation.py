import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict

SCORING = {
    "ap": "average_precision", "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy",
}


def validate_probability(probability):
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("概率必须是一维、非空、有限数组")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("概率必须在 [0,1]")
    return p


def validate_threshold(threshold):
    try:
        t = float(threshold)
    except (ValueError, TypeError) as exc:
        raise ValueError("阈值必须是数值") from exc
    if not np.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("阈值必须有限且在 [0,1]")
    return t


def _check_xy(X, y):
    if len(X) != len(y) or set(np.asarray(y)) != {0, 1}:
        raise ValueError("X/y 必须等长，且 y 包含 0/1 两类")
    if hasattr(X, "index") and hasattr(y, "index") and not X.index.equals(y.index):
        raise ValueError("X/y index 未对齐")


def _check_folds(folds, n):
    coverage = np.zeros(n, dtype=int)
    universe = set(range(n))
    for train, valid in folds:
        train, valid = np.asarray(train), np.asarray(valid)
        if not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(valid.dtype, np.integer):
            raise ValueError("fold 索引必须为整数")
        tr, va = set(train), set(valid)
        if (not tr or not va or tr & va or tr | va != universe
                or len(tr) != len(train) or len(va) != len(valid)):
            raise ValueError("fold 必须是完整、无重复且不相交的划分")
        coverage[valid] += 1
    if not (coverage == 1).all():
        raise ValueError("每位客户必须恰验证一次")


def make_cv_splits(X, y, *, n_splits=5, random_state=42):
    _check_xy(X, y)
    if np.bincount(np.asarray(y, dtype=int)).min() < n_splits:
        raise ValueError("每个类别的样本量必须不少于折数")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(cv.split(X, y))
    _check_folds(folds, len(X))
    return folds


def evaluate_cv_pipeline(model_name, pipeline, X, y, cv_splits):
    _check_xy(X, y)
    _check_folds(cv_splits, len(X))
    scores = cross_validate(pipeline, X, y, cv=cv_splits, scoring=SCORING,
                            n_jobs=1, error_score="raise", return_train_score=False)
    row = {"model": model_name}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_std"] = float(values.std(ddof=1))
    return row, scores


def generate_oof_probability(pipeline, X, y, cv_splits):
    _check_xy(X, y)
    _check_folds(cv_splits, len(X))
    # y 已验证为 0/1；sklearn 按类别排序返回列，第二列对应 1。
    probability = cross_val_predict(clone(pipeline), X, y, cv=cv_splits,
                                    method="predict_proba", n_jobs=1)[:, 1]
    return validate_probability(probability)


def metrics_at_threshold(y_true, probability, threshold):
    p, t = validate_probability(probability), validate_threshold(threshold)
    y = np.asarray(y_true)
    if y.ndim != 1 or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("目标必须是一维 0/1 且与概率等长")
    predicted = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": t, "recall": float(recall_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "selected_count": int(predicted.sum()), "selected_share": float(predicted.mean()),
    }


def select_capacity_threshold(probability, *, capacity_share=0.20):
    p = validate_probability(probability)
    share = float(capacity_share)
    if not np.isfinite(share) or not 0 < share <= 1:
        raise ValueError("容量比例必须在 (0,1]")
    capacity = int(np.floor(share * len(p)))
    levels, counts = np.unique(p, return_counts=True)
    levels, counts = levels[::-1], counts[::-1]
    eligible = np.flatnonzero(np.cumsum(counts) <= capacity)
    if len(eligible) == 0:
        raise ValueError("没有非空的分数组满足容量")
    return float(levels[eligible[-1]])
