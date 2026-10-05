"""读取可信模型，对新客户评分；不训练任何步骤。"""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import joblib
from sklearn.utils.validation import check_is_fitted
from .config import ID_COLUMN, FEATURE_COLUMNS, TARGET_MAPPING, ARTIFACT_VERSION
from .data import load_customers, validate_customers, prepare_features
from .evaluation import validate_probability, validate_threshold

VERSION_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib")


def runtime_versions():
    """学习版采用保守的精确版本匹配，避免误加载不同环境训练的产物。"""
    return {"python": platform.python_version(), **{p: version(p) for p in VERSION_PACKAGES}}


def load_artifacts(model_path, metadata_path):
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata 必须是 JSON 对象")
    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("artifact 格式版本不匹配")
    if metadata.get("feature_columns") != FEATURE_COLUMNS or metadata.get("target_mapping") != TARGET_MAPPING:
        raise ValueError("schema 或目标映射不匹配")
    validate_threshold(metadata.get("threshold"))
    if metadata.get("versions") != runtime_versions():
        raise ValueError("运行环境版本与训练环境不一致，请恢复记录的版本")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != metadata.get("model_sha256"):
        raise ValueError("模型文件哈希与 metadata 不匹配")
    # 哈希匹配不是安全认证；调用者仍只能提供可信来源的 joblib。
    pipeline = joblib.load(model_path)
    check_is_fitted(pipeline.named_steps["model"])
    if list(pipeline.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("模型学习的字段顺序不匹配")
    if set(pipeline.classes_) != {0, 1}:
        raise ValueError("模型类别必须为 0 和 1")
    return pipeline, metadata


def score_customers(customers, pipeline, threshold):
    """核心函数三：准备特征 -> 概率 -> 阈值分类 -> 对齐客户 ID。"""
    validate_customers(customers)
    t = validate_threshold(threshold)
    features = prepare_features(customers)
    check_is_fitted(pipeline.named_steps["model"])
    classes = np.asarray(pipeline.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("模型必须使用 0/1 类别")
    positive_column = int(np.flatnonzero(classes == 1)[0])
    probability = validate_probability(pipeline.predict_proba(features)[:, positive_column])
    if len(probability) != len(customers):
        raise ValueError("概率行数与客户行数不一致")
    predicted = (probability >= t).astype(int)
    # numpy 数组按位置对齐，避免不同 Series index 造成意外错位。
    return pd.DataFrame({
        ID_COLUMN: customers[ID_COLUMN].astype("string").to_numpy(),
        "churn_probability": probability,
        "predicted_class": predicted,
        "risk_tier": np.where(predicted == 1, "High", "Standard"),
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score customer CSV using a trusted frozen pipeline.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/churn_pipeline.joblib"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/model_metadata.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        protected = [args.input, args.model, args.metadata]
        if any(args.output.resolve() == p.resolve() for p in protected):
            raise ValueError("输出不能覆盖输入、模型或元数据文件")
        if args.output.exists() and not args.overwrite:
            raise ValueError("输出已存在；确认后使用 --overwrite")
        customers = load_customers(args.input)
        pipeline, metadata = load_artifacts(args.model, args.metadata)
        extra = sorted(set(customers.columns) - {ID_COLUMN, *FEATURE_COLUMNS})
        if extra:
            print(f"忽略额外字段: {extra}")
        result = score_customers(customers, pipeline, metadata["threshold"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w" if args.overwrite else "x", encoding="utf-8", newline="") as handle:
            result.to_csv(handle, index=False)
        print(f"已保存 {len(result)} 行: {args.output}")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"预测失败: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
