# Model card — Telco retention-review demo

## Purpose and boundary

Version: portfolio v1, frozen experiment. Intended readers are reviewers and
analysts evaluating a reproducible customer-risk prioritization workflow.
The score supports a **candidate list for human review**, not automatic
contact, eligibility decisions, credit decisions or evidence of retention.

Data source: [IBM Telco sample repository](https://github.com/IBM/telco-customer-churn-on-icp4d).
Retrieval, checksum and non-redistribution policy: [data README](../data/README.md).
The sample represents a fictional telecommunications population, not bank
customers. There are 7,043 rows, 5,174 `No` and 1,869 `Yes` labels. Each row is
one customer snapshot. `Churn` is the supplied observed label, not this
project's prediction. The table cannot establish a feature cutoff and future
outcome window; the hypothetical bank's 90-day definition is not a Telco claim.

## Inputs and processing

`customerID` identifies output rows and is excluded from features. The target
mapping is `No → 0`, `Yes → 1`. The 19 inputs are:

- Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`.
- Categorical: `gender`, `SeniorCitizen`, `Partner`, `Dependents`,
  `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`,
  `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`,
  `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.

Blank numeric values become missing values; malformed nonblank numeric text
and infinities are rejected. In particular, the original file has 11 blank
`TotalCharges` records. Numeric medians and categorical most-frequent values
are learned from each training fold, not the full table. Categories use
one-hot encoding with unknown categories ignored. Logistic Regression uses
numeric scaling; XGBoost does not need it. The ID and any extra outcome-like
columns are excluded by the feature whitelist. Prediction never refits.

Including demographic fields in a public experiment is **not approval** to
use them operationally. No subgroup performance/fairness audit has been
completed. Future use requires justification and review of fields, proxies,
subgroup errors and contact policies.

## Validation and selection

1. One stratified 80/20 random split, seed 42: 5,634 train, 1,409 holdout.
2. Five common stratified training folds, shuffled with seed 42. Every fitted
   preprocessing step stays inside each model pipeline and fold.
3. Compare Dummy and Logistic Regression with six fixed XGBoost candidates
   by mean average precision (AP). AP is not trapezoidal PR AUC.
4. Select XGBoost `B_shallow`; derive its out-of-fold scores on training rows.
5. Choose the threshold under an illustrative 20% training review capacity,
   retaining whole equal-score groups rather than splitting tied scores.
6. Fit the selected pipeline on training rows only; evaluate holdout once
   after choices are frozen. Later engineering reproductions repeat those
   choices; they do not create a new independent holdout.

The selected XGBoost configuration is `n_estimators=300`,
`learning_rate=0.05`, `max_depth=2`, `min_child_weight=1`, `subsample=0.8`,
`colsample_bytree=0.8`, `reg_lambda=1`, `gamma=0`, `tree_method=hist`,
`objective=binary:logistic`, `eval_metric=logloss`, seed 42, one model thread.

| Model | Training CV AP, mean ± SD | CV ROC AUC, mean | CV recall at 0.5 |
|---|---:|---:|---:|
| Dummy | 0.265353 ± 0.000105 | 0.500000 | 0.000000 |
| Logistic Regression | 0.661436 ± 0.021602 | 0.846128 | 0.543144 |
| XGBoost B_shallow | 0.670135 ± 0.021588 | 0.849286 | 0.533110 |

Paired fold AP improves by 0.008699 on average (SD 0.005601); all five fold
differences are positive. This is a small ranking improvement, not a proven
business benefit or a significance test. Logistic Regression remains a
credible simpler alternative. Winning-candidate CV estimates are optimistic
because the same folds were used to compare candidates; nested validation
was not performed. Fold SD is descriptive, not a confidence interval.

## Frozen decision threshold and holdout

Threshold: `0.517093300819397`, not its rounded display value.
The training OOF list contains 1,126 / 5,634 customers (19.9858%), with
precision 67.3179% and recall 50.7023%. The capacity is a **scenario assumption**,
not a bank-approved operating requirement or a profit-optimal threshold.

| Metric | Holdout value |
|---|---:|
| Average precision | 0.666219 |
| ROC AUC | 0.846890 |
| Precision | 0.680147 |
| Recall | 0.494652 |
| F1 | 0.572755 |
| Accuracy | 0.804116 |
| Selected customers | 272 / 1,409 (19.3045%) |

| Actual label | Predicted Standard | Predicted High |
|---|---:|---:|
| No churn | TN 948 | FP 87 |
| Churn | FN 189 | TP 185 |

False negatives miss potential retention-review opportunities; false
positives consume review/contact capacity and may inconvenience customers.
The model misses 189 of 374 holdout churners. Neither the 185 true positives
nor the 272 selected customers are a count of customers saved. A fixed
threshold can exceed 20% on a different batch; a hard batch capacity would
require a separately specified ranking/tie policy. Do not retune on holdout.

## Explanations

The training command produces global and single-customer TreeSHAP plots.
It samples up to 200 training rows with seed 42, uses XGBoost's
`pred_contribs=True`, and verifies that contributions plus the base value
sum to the raw model margin. `shap` handles visualization; native XGBoost
contributions avoid a TreeExplainer compatibility issue with this runtime.
The local plot uses the first sampled row and records its source index.
Values are log-odds contributions for transformed features, not causal
effects or direct probability changes. Related predictors and one-hot
levels can share contribution; the 200-row view is not a population audit.

## Known limitations and safeguards

- Full-data EDA preceded the modeling split. This is analyst-level
  contamination even though preprocessing is fitted correctly. Holdout
  results should be described as limited demo evidence, not a pristine
  prospective validation.
- The static table does not prove that every field was available before
  churn. A whitelist prevents accidental extra target inputs, but cannot
  establish historical availability. Banking needs point-in-time joins and
  historical feature snapshots.
- No timestamps support rolling/time-based evaluation, label maturation or
  real deployment drift assessment. No conclusions about causation,
  intervention uplift, financial return or transfer to banking are supported.
- No probability calibration or subgroup fairness validation is established.
  `High` is a model threshold flag, not an instruction to contact a customer.
- Unknown categories are accepted technically, but frequent unknowns,
  missingness, score shifts and selection-rate shifts would require review.
- Real-world deployment would need an accountable business/data owner,
  permissioned and minimized input access, masking, audit logs, approved
  contact exclusions and human review. None is demonstrated by this demo.
- Saved joblib files must come from a trusted source. Metadata stores schema,
  library/Python versions and hashes; checks detect mismatch, not trust.

The reference run used Python 3.10.7 and the versions in
[`constraints-py310.txt`](../constraints-py310.txt). Retraining writes its own
exact runtime metadata. Different package/platform combinations are not
guaranteed to reproduce identical floating-point scores. The training CLI
checks the frozen data hash, winning XGBoost candidate and OOF threshold.

Measured values above come from the saved reference run's
`model_comparison.csv`, `paired_ap.csv`, `oof_metrics.json` and
`test_metrics.json`; rerun training to generate these locally. Generated model
binaries and customer-level CSVs are not public source artifacts.
