"""原始记录 -> 校验 -> 确定性转换；统计量留给 Pipeline 学习。"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import ID_COLUMN, TARGET_COLUMN, FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_MAPPING


def load_customers(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 CSV 文件: {path}")
    # pandas 可能自动改写重复表头，因此读取 DataFrame 前先检查原始表头。
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header:
        raise ValueError("CSV 没有表头")
    if len(header) != len(set(header)):
        raise ValueError("CSV 存在重复列名")
    return pd.read_csv(path, dtype={ID_COLUMN: "string"}, encoding="utf-8-sig")


def _convert_numeric(series, *, column_name):
    text = series.astype("string").str.strip()
    missing = series.isna() | text.eq("").fillna(False)
    converted = pd.to_numeric(series.mask(missing, np.nan), errors="coerce")
    invalid = (~missing) & converted.isna()
    if invalid.any():
        raise ValueError(f"{column_name} 含无法解析的非空数值")
    converted = converted.astype(float)
    if np.isinf(converted.to_numpy()).any():
        raise ValueError(f"{column_name} 含无穷值")
    return converted


def validate_customers(customers, *, require_target=False):
    if not isinstance(customers, pd.DataFrame) or customers.empty:
        raise ValueError("输入必须是非空 DataFrame")
    if not customers.columns.is_unique:
        raise ValueError("列名重复")
    required = [ID_COLUMN, *FEATURE_COLUMNS]
    if require_target:
        required.append(TARGET_COLUMN)
    missing = [c for c in required if c not in customers.columns]
    if missing:
        raise ValueError(f"缺少字段: {missing}")
    ids = customers[ID_COLUMN].astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("customerID 必须非空且唯一")
    for column in NUMERIC_FEATURES:
        _convert_numeric(customers[column], column_name=column)
    if require_target:
        target = customers[TARGET_COLUMN]
        if target.isna().any() or not target.isin(TARGET_MAPPING).all():
            raise ValueError("Churn 必须为 Yes 或 No")


def prepare_features(customers):
    """核心函数一：白名单选列、转换类型、返回独立副本。"""
    validate_customers(customers)
    features = customers.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        features[column] = _convert_numeric(features[column], column_name=column)
    for column in CATEGORICAL_FEATURES:
        # 先用可空字符串处理空白，最后转回 object + np.nan 交给 sklearn。
        values = features[column].astype("string").str.strip()
        if column == "SeniorCitizen":
            values = values.replace({"0.0": "0", "1.0": "1"})
        values = values.mask(values.eq(""), pd.NA)
        features[column] = values.astype(object).where(values.notna(), np.nan)
    return features


def split_customers(customers, *, random_state=42, test_size=0.20):
    validate_customers(customers, require_target=True)
    X = prepare_features(customers)
    y = customers[TARGET_COLUMN].map(TARGET_MAPPING).astype(int)
    ids = customers[ID_COLUMN].astype("string").copy()
    if y.nunique() != 2:
        raise ValueError("训练数据需要同时包含两个类别")
    return tuple(train_test_split(
        X, y, ids, test_size=test_size, stratify=y, random_state=random_state,
    ))
