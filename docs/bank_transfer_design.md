# Hypothetical bank-transfer requirements

This document is a **design exercise**, not a description of a bank system,
approved work order or experiment using bank data. The actual internship
connection is interpreting internal work orders, SQL extraction of requested
customer/transaction data, grain and field checks, and cross-department
requirements coordination. The Telco model would not simply be reused on bank
columns: a banking dataset, label and model require separate development.

## 1. One objective and one observation unit

Proposed objective: predict whether an eligible customer will voluntarily
close at least one eligible account within 90 days after a prediction cutoff.
This is **account-closure risk at customer grain**, not proof that the entire
customer relationship will end. Closing one of several accounts can be
routine consolidation. The business owner must confirm this interpretation
before extraction; total-relationship exit would be a different label.

| Requirement | Proposed v1 definition |
|---|---|
| Observation unit | One customer at one cutoff: `(customer_key, cutoff_date)` |
| Population | Customers with ≥1 eligible open retail deposit account at cutoff |
| Feature window | Six calendar months before cutoff: `[cutoff − 6 months, cutoff)` |
| Prediction moment | Start of cutoff day; use information available before it |
| Outcome window | `[cutoff, cutoff + 90 days)` |
| Positive label | At least one account open/eligible at cutoff subsequently closes voluntarily in that window |
| Negative label | Full 90-day follow-up is available and no qualifying closure occurs |
| Censoring | Insufficient follow-up is not a negative; exclude from supervised labels |
| Intended output | Masked customer key, cutoff, score, review flag and model version |

These are proposed definitions, not confirmed bank rules. Product scope,
calendar/time-zone conventions, closure reason codes and joint-account
attribution require sign-off. A reasonable initial restriction is individually
owned accounts; joint accounts need an explicit ownership/attribution policy.

## 2. Eligibility and exclusions

Only use status known at cutoff for the scoring population. Proposed exclusions
include already closed accounts, test/employee records where policy requires,
customers with no eligible accounts, known death/incapacity records, and
known fraud/legal/administrative closure processes. Contact suppression and
consent rules are a separate filter for any eventual outreach list.

For labels, exclude non-retention closure events such as fraud-related,
administrative, legal, death/accident-related or duplicate-account closures
where the confirmed reason indicates a non-voluntary event. Unknown reasons
must not silently become voluntary positives or healthy negatives.
Pre-specify how these competing events affect follow-up (e.g., exclude the
affected observation from training labels and report counts). If a qualifying
voluntary closure already occurred before a competing event, retain that
positive under the agreed event ordering rule.

Critically, a non-voluntary closure discovered **after** cutoff may be used
for outcome construction, but must not retrospectively alter the as-of
scoring eligibility or become a feature. Report the resulting selection bias
and censored/excluded counts. A person who becomes ineligible for contact is
suppressed at action time regardless of their earlier risk score.

## 3. Data request and SQL-ready extraction contract

All table and key names below are conceptual placeholders, not internal bank
schema. Request definitions and approved lineage rather than confidential
table extracts or employee details for a portfolio.

| Domain / source grain | Keys and as-of rule | Customer-level features or purpose |
|---|---|---|
| Customer history | `customer_key`; effective interval contains cutoff, record available before cutoff | Relationship tenure, permitted segment; eligibility |
| Account and ownership history | `account_key`, `customer_key`; ownership valid at cutoff | Eligible account count, account tenure; customer/account mapping |
| Product holdings | Account/customer + product + effective dates | Product counts and types known at cutoff |
| Daily balances | Account + business date; window and publication time before cutoff | Window mean/min balance, last available balance, recent change |
| Transactions | Unique transaction key + account + event and availability time | Counts/amounts by direction and approved category, recency; avoid raw descriptions |
| Channel activity | Customer/session/event keys + timestamp | Activity frequency, last activity date, inactivity indicators |
| Service and complaints | Customer/case keys; opened/updated availability times | Prior case counts, unresolved-at-cutoff status, recency |
| Closure outcomes | Account + effective closure date + reason + observation completeness | Future label only; never join into the feature table |

Suggested extraction sequence:

1. Build the cutoff population and as-of ownership mapping. Enforce one row
   per `(customer_key, cutoff_date)` and document account-to-customer cardinality.
2. Filter each fact domain to its allowed window **and** availability time.
   A transaction backdated into the window but loaded after cutoff was not
   available to the model. Reconstructed historical complaint status must
   not use the latest post-cutoff resolution.
3. Aggregate each domain separately to customer-cutoff grain before joining.
   Joining raw transactions directly to raw complaints creates many-to-many
   row multiplication and incorrect sums.
4. Left-join aggregates to the eligible population; distinguish a true zero
   activity count from an absent or delayed source extract. Preserve flags
   for data completeness instead of imputing source outages as zero business.
5. Construct labels in a separate query using only matured outcome windows
   and the approved event exclusions. Join labels only for supervised training.
6. Reconcile row/ID counts, uniqueness, date bounds, exclusions, unmatched
   accounts and sampled aggregate totals to the source. Deliver a dictionary,
   extraction version/cutoff, QA report and documented unresolved exceptions.

No executable SQL is supplied without an actual authorized schema. The
contract is detailed enough to discuss keys, time predicates and aggregation
with a data owner without inventing bank tables.

## 4. Ownership and governance checkpoints

| Checkpoint | What must be confirmed before real work |
|---|---|
| Business owner | Who approves the closure definition, population, purpose, contact capacity and exclusions? |
| Source/data owner | Who can explain the field, update lag, history and approved extraction path for each domain? |
| Permission | Does the analyst/service account have approved access for this use and these fields? Technical access alone is not business approval. |
| Masking | Replace direct identity/contact details with an approved surrogate key. Keep any re-identification mapping in a separately controlled system. |
| Minimum necessary | Request only justified columns, population, date range and recipients; prefer approved aggregates over unnecessary raw transaction text. |
| QA and lineage | Record grain, joins, availability times, exclusions, missingness, reconciliations and extraction version. |
| Audit and retention | Log approvals and deliveries; confirm storage location, access, retention period and deletion process under applicable internal policy. |
| Review and action | Risk/compliance/business reviewers approve permitted use, contact suppressions and human review before outreach. |

Named employees and private records are not required for this portfolio.
These are questions a real project would resolve through the bank's approved
process, not a request for acquaintances to disclose internal information.

## 5. Validation and business testing

Use earlier cutoffs for training, later ones for validation and the newest
fully matured interval for final testing. Training labels must be available
before the next simulated model-fit date; leave at least the label horizon
where needed to prevent outcome-window overlap across the boundary. Fit
preprocessing only on training data, and freeze model/threshold choices
before the final test.

Repeated customer snapshots need deliberate handling. For an existing-customer
use case, the same customer may appear later, but no future records may enter
earlier training features. Report customer overlap and account for correlated
rows when estimating uncertainty. Add a customer-disjoint temporal evaluation
if claiming generalization to previously unseen customers; a random row split
does not establish either temporal or new-customer performance.

Compare a no-skill baseline, Logistic Regression and a nonlinear candidate on
common permitted folds. Review AP, precision/recall at approved workload,
false negatives, contact eligibility, subgroup errors, calibration and
data/score drift. Determine thresholds with validation data and explicit
capacity/cost assumptions, never future test outcomes.

Before claiming business retention impact, conduct an approved randomized
intervention/holdout experiment or another defensible causal design. Measure
incremental voluntary closures avoided, customer complaints, contact costs
and net benefit. High predicted risk does not imply high treatment benefit;
some high-risk customers will leave despite contact, while some would stay
without it. Telco classification metrics cannot establish bank ROI.
