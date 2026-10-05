"""复现已冻结的实验；不依据 holdout 调整模型或阈值。"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from .config import (
    FEATURE_COLUMNS, TARGET_MAPPING, RANDOM_STATE, SELECTED_MODEL,
    FROZEN_THRESHOLD, CAPACITY_SHARE, DATA_SHA256, ARTIFACT_VERSION, XGB_CANDIDATES,
)
from .data import load_customers, split_customers
from .models import build_model_pipeline
from .evaluation import (
    make_cv_splits, evaluate_cv_pipeline, generate_oof_probability,
    select_capacity_threshold, metrics_at_threshold,
)
from .predict import score_customers, runtime_versions


def save_artifacts(pipeline, metadata, output_dir):
    folder = Path(output_dir) / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / "churn_pipeline.joblib"
    metadata_path = folder / "model_metadata.json"
    if model_path.exists() or metadata_path.exists():
        raise FileExistsError("模型产物已存在，请选择新输出目录")
    joblib.dump(pipeline, model_path)
    record = {**metadata, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    metadata_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return model_path, metadata_path


def save_figures(pipeline, X_train, y_holdout, probability, threshold, table, report_dir):
    # CLI 保存图片，不需要桌面窗口；import 模块本身不会绘图。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from xgboost import DMatrix
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

    def save_current(name):
        plt.gcf().savefig(report_dir / name, bbox_inches="tight", dpi=150)
        plt.close("all")

    table.plot(x="threshold", y=["recall", "precision", "selected_share"], ylim=(0, 1))
    save_current("threshold_tradeoff.png")
    ConfusionMatrixDisplay.from_predictions(y_holdout, (probability >= threshold).astype(int), labels=[0, 1])
    save_current("confusion_matrix.png")
    PrecisionRecallDisplay.from_predictions(y_holdout, probability)
    save_current("pr_curve.png")
    RocCurveDisplay.from_predictions(y_holdout, probability)
    save_current("roc_curve.png")

    sample = X_train.sample(n=min(200, len(X_train)), random_state=RANDOM_STATE)
    prep = pipeline.named_steps["preprocess"]
    transformed = prep.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    booster = pipeline.named_steps["model"].get_booster()
    matrix = DMatrix(transformed)
    contributions = booster.predict(matrix, pred_contribs=True)
    margin = booster.predict(matrix, output_margin=True)
    if not np.allclose(contributions.sum(axis=1), margin, atol=1e-5):
        raise ValueError("SHAP 加和检查失败")
    explanation = shap.Explanation(
        values=contributions[:, :-1], base_values=contributions[:, -1],
        data=transformed, feature_names=list(prep.get_feature_names_out()),
    )
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    save_current("shap_global.png")
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    save_current("shap_local.png")
    # 留下样本来源，方便解释图中的客户选择和 log-odds 单位。
    explanation_record = {"sample_seed": RANDOM_STATE, "rows": len(sample),
                          "local_position": 0, "local_source_index": str(sample.index[0]),
                          "unit": "raw margin / log-odds"}
    (report_dir / "explanation_metadata.json").write_text(json.dumps(explanation_record, indent=2), encoding="utf-8")


def run_training(input_path, output_dir):
    input_path, output_dir = Path(input_path), Path(output_dir)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError("数据哈希与冻结实验不一致；不要套用旧阈值重现新数据")
    for name in ("artifacts", "reports"):
        destination = output_dir / name
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(f"输出目录已有内容: {destination}")
    customers = load_customers(input_path)
    X_train, X_holdout, y_train, y_holdout, id_train, id_holdout = split_customers(customers)
    folds = make_cv_splits(X_train, y_train)
    models = {"Dummy": build_model_pipeline("dummy"),
              "Logistic Regression": build_model_pipeline("logistic")}
    models.update({name: build_model_pipeline("xgboost", xgb_params=params)
                   for name, params in XGB_CANDIDATES.items()})
    rows, scores = [], {}
    for name, pipeline in models.items():
        row, cv_scores = evaluate_cv_pipeline(name, pipeline, X_train, y_train, folds)
        rows.append(row)
        scores[name] = cv_scores
    results = pd.DataFrame(rows).set_index("model")
    winner = results.loc[list(XGB_CANDIDATES), "ap_mean"].idxmax()
    if winner != SELECTED_MODEL:
        raise ValueError("XGBoost winner 与冻结记录不一致，需诊断环境而非自动换模型")
    oof = generate_oof_probability(models[SELECTED_MODEL], X_train, y_train, folds)
    reproduced_threshold = select_capacity_threshold(oof, capacity_share=CAPACITY_SHARE)
    if not np.isclose(reproduced_threshold, FROZEN_THRESHOLD, atol=1e-6, rtol=0):
        raise ValueError("OOF 阈值与冻结记录不一致")
    threshold = FROZEN_THRESHOLD  # 用保存的完整值，不因 holdout 改动。
    threshold_table = pd.DataFrame([
        metrics_at_threshold(y_train, oof, t)
        for t in np.round(np.arange(.05, 1, .05), 2)
    ])
    pipeline = clone(models[SELECTED_MODEL]).fit(X_train, y_train)
    probability = pipeline.predict_proba(X_holdout)[:, 1]
    holdout_metrics = metrics_at_threshold(y_holdout, probability, threshold)
    holdout_metrics.update(average_precision=float(average_precision_score(y_holdout, probability)),
                           roc_auc=float(roc_auc_score(y_holdout, probability)))
    metadata = {
        "artifact_version": ARTIFACT_VERSION, "model": SELECTED_MODEL,
        "threshold": threshold, "seed": RANDOM_STATE, "feature_columns": FEATURE_COLUMNS,
        "target_mapping": TARGET_MAPPING, "xgb_parameters": XGB_CANDIDATES[SELECTED_MODEL],
        "versions": runtime_versions(), "data_sha256": digest,
        "capacity_share_assumption": CAPACITY_SHARE,
        "selection_source": "Frozen training CV/OOF decision; holdout used for evaluation only",
        "limitations": "Same-fold selection optimism; earlier full-data EDA; no causal or bank ROI claims",
    }
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    results.loc[["Dummy", "Logistic Regression", SELECTED_MODEL]].to_csv(report_dir / "model_comparison.csv")
    results.loc[list(XGB_CANDIDATES)].to_csv(report_dir / "xgb_candidates.csv")
    pd.DataFrame({"fold": range(1, 6), "logistic_ap": scores["Logistic Regression"]["test_ap"],
                  "xgboost_ap": scores[SELECTED_MODEL]["test_ap"]}).to_csv(report_dir / "paired_ap.csv", index=False)
    threshold_table.to_csv(report_dir / "threshold_table.csv", index=False)
    for name, values in [("test_metrics.json", holdout_metrics),
                         ("oof_metrics.json", metrics_at_threshold(y_train, oof, threshold))]:
        (report_dir / name).write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")
    sample = score_customers(customers.loc[X_holdout.index], pipeline, threshold)
    if sample["customerID"].tolist() != id_holdout.tolist():
        raise ValueError("预测输出 ID 对齐失败")
    sample.to_csv(report_dir / "prediction_sample.csv", index=False)
    save_figures(pipeline, X_train, y_holdout, probability, threshold, threshold_table, report_dir)
    model_path, metadata_path = save_artifacts(pipeline, metadata, output_dir)
    return {"model": model_path, "metadata": metadata_path, "reports": report_dir}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce the frozen Telco experiment.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/customer_churn.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        paths = run_training(args.input, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"训练失败: {exc}\n")
    print({name: str(path) for name, path in paths.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
