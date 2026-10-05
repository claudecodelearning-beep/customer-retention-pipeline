# Dataset

## Raw data

- File: `raw/customer_churn.csv`
- Dataset: IBM Telco Customer Churn
- Source repository: https://github.com/IBM/telco-customer-churn-on-icp4d
- Direct source file: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Retrieved: 2026-08-02
- SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

## Shape and target

- Rows: 7,043 customers
- Columns: 21
- Target: `Churn`
- Target distribution: 5,174 `No`; 1,869 `Yes`

## Project usage note

This is a public, fictional telecommunications dataset. It may be used to build
an industry-inspired customer-retention pipeline, but it must not be described
as Bank of China data, real banking customer data, or a production banking
dataset. Any financial-services framing in the portfolio should be presented as
a transferable workflow or a public-data prototype.

Processed outputs should be written to `processed/` and the raw CSV should
remain unchanged.

## Download for reproduction

Run these commands from the project root, only when the destination file does
not already exist. Download to a temporary name first; move it into place
after the checksum matches. Do not replace a different local dataset blindly.

```bash
mkdir -p data/raw
curl -fL --max-time 60 \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv \
  -o data/raw/customer_churn.download.csv
python -c 'import hashlib; from pathlib import Path; p=Path("data/raw/customer_churn.download.csv"); expected="16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"; actual=hashlib.sha256(p.read_bytes()).hexdigest(); print(actual); assert actual == expected, "Dataset checksum mismatch"'
mv -n data/raw/customer_churn.download.csv data/raw/customer_churn.csv
```

Do not run the final move if verification fails. The download is approximately
948 KiB. The source branch can theoretically change; the checksum, not the URL
alone, defines this project's exact input bytes. Do not normalize line endings
or resave the CSV before running the frozen experiment.

## Attribution and publication policy

The original [IBM repository](https://github.com/IBM/telco-customer-churn-on-icp4d)
includes a [repository license](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE).
This project links to that source and does **not** bundle or relicense the raw
data. Check source terms before any separate redistribution. A source
repository's license is not a license chosen for this project's own code.
The public `.gitignore` excludes `data/raw/`, `data/processed/` and generated
CSV files. Source attribution is retained even though the data is downloaded
locally rather than committed.
