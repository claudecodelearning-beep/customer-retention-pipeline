"""字段和冻结的实验配置；这里只保存规则，不读取数据。"""

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.20
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [c for c in FEATURE_COLUMNS if c not in NUMERIC_FEATURES]
TARGET_MAPPING = {"No": 0, "Yes": 1}
SELECTED_MODEL = "B_shallow"
FROZEN_THRESHOLD = 0.517093300819397
CAPACITY_SHARE = 0.20  # 假设审核容量，不是实测银行容量。
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
ARTIFACT_VERSION = 1
BASE_XGB_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
    "min_child_weight": 1, "subsample": 0.80, "colsample_bytree": 0.80,
    "reg_lambda": 1.0, "gamma": 0.0,
}
# **base 先展开字典，后面的同名字段覆盖默认值，不修改 base 本身。
XGB_CANDIDATES = {
    "A_base": dict(BASE_XGB_PARAMS),
    "B_shallow": {**BASE_XGB_PARAMS, "max_depth": 2},
    "C_deeper": {**BASE_XGB_PARAMS, "max_depth": 4},
    "D_slow_learning": {**BASE_XGB_PARAMS, "n_estimators": 500, "learning_rate": 0.03},
    "E_fast_learning": {**BASE_XGB_PARAMS, "n_estimators": 200, "learning_rate": 0.10},
    "F_regularized": {
        **BASE_XGB_PARAMS, "min_child_weight": 5, "reg_lambda": 5.0, "gamma": 0.10,
    },
}
