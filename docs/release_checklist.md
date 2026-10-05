# Local-to-public release checklist

Local reproduction and learning checkpoints. Public notebook copies contain
code and final explanations; original drafts remain local. No license has
been selected for the project code. Publication does not imply learner sign-off.

## Reproduction evidence

- [x] Install the project and reference constraints in a new Python 3.10 environment;
  run `pip check` and all tests.
- [x] Retrieve the public CSV and match the checksum in `data/README.md`.
- [x] Run full training into an empty directory, then run batch prediction.
  Verify four output columns, ID alignment and probability bounds.
- [x] Reconcile README/model-card figures with generated CV and holdout reports.
- [x] Execute all three notebooks top to bottom with fresh kernels, without
  overwriting the original learning drafts. Investigate any saved-error or
  execution failure; do not hide it using `--allow-errors`.

Local acceptance on 2026-10-02: a new macOS ARM64 Python 3.10.7 virtual
environment installed the constrained project; `pip check` passed, and all
22 tests passed. The downloaded CSV matched the frozen hash. Full training
reproduced the reference metric JSONs exactly, and the prediction contract
passed for all 7,043 rows. Audit/EDA/modeling notebooks executed 20/45/69
nonempty code cells respectively, with no error outputs and unchanged original
source hashes. The first sandboxed notebook run hit a process-cleanup
permission error; an authorized unrestricted rerun removed that error.
Jupyter still emitted its local TCP transport warning; this check is not a
network-security certification. Windows and Linux have not been locally
validated, and the remote CI checkbox below remains intentionally open.

## Publication boundary

- [x] Review selected notebook source, output, metadata and any absolute paths;
  retain the learner's process locally. Make only agreed changes to public copies.
- [x] Review the exact staging list and staged diff for secrets, personal
  details, local paths and unintended files. `.gitignore` is not a secret scanner.
- [x] Keep agent instructions, learning guide, planning documents and backups
  local. Ignore venv/caches, raw data, generated prediction lists and binaries.
- [x] Use the documented data download instead of republishing the raw CSV.
  Confirm any intended redistribution separately against the source terms.
- [ ] Decide licensing for this project's own code before representing it as
  licensed open-source software; no license is selected on the author's behalf.
- [x] The author authorized a public repository at claudecodelearning-beep/customer-retention-pipeline.
- [ ] After pushing, verify the actual GitHub Actions run is green. Local
  tests or a syntactically valid workflow do not satisfy this remote gate.

## Learner sign-off

- [ ] Explain why customerID and Churn are not model inputs, and why filling
  medians before splitting would leak information.
- [ ] Explain the small AP gain over Logistic Regression and selection optimism.
- [ ] Explain 20% capacity, the frozen threshold, 185 TP / 87 FP / 189 FN,
  and why a future batch may have a different selected fraction.
- [ ] Explain what SHAP describes and why it does not prove intervention effects.
- [ ] Distinguish real internship tasks, this independent Telco demo and the
  hypothetical bank-transfer design.

The CI job uses synthetic test fixtures, so it does not need private records
or download the training CSV. The full frozen experiment and notebook checks
are separate from that lightweight CI gate.
