# Setup & Run Guide

## 1. Requirements
- Python 3.10+ (3.11/3.12 also fine)
- pip
- ~2 GB free disk (XGBoost + SHAP + MLflow pull in some size)

## 2. Install

```bash
cd loan-approval-risk-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or with `make`:
```bash
make setup
```

## 3. Get the dataset — you don't have to, to start

The repo ships with a **synthetic** data generator, so there's nothing to
download to see the whole pipeline run. The first time you train, if
`data/loan_applications.csv` doesn't exist, 6,000 synthetic applicants are
generated automatically from the risk model in `src/data_ingestion.py`.

### Using real data instead (recommended for a stronger portfolio piece)

1. Download a loan dataset, e.g. from Kaggle:
   - "Loan Prediction Problem Dataset"
   - "Lending Club Loan Data"
2. Rename/remap its columns to match this exact schema:
   ```
   applicant_income, coapplicant_income, loan_amount, loan_term_months,
   employment_years, age, dependents, previous_defaults,
   existing_loans_count, credit_score, education, self_employed,
   property_area, marital_status, loan_approved
   ```
   (If your real dataset uses different column names or fewer features,
   edit the `columns:` section of `config.yaml` to match instead of
   forcing your data into this exact shape.)
3. Save it as `data/loan_applications.csv`, overwriting the synthetic file.
4. Run `make train`. The schema validator (`validate_schema` in
   `src/data_ingestion.py`) checks required columns, target binarity, and
   missing-value thresholds before any model training starts — so a bad
   column mapping fails fast with a clear error instead of silently
   training garbage.

## 4. Train

```bash
python train.py --config config.yaml
```
or
```bash
make train
```

This runs, in order:
1. Data ingestion + schema validation
2. Feature engineering (income-to-loan ratio, debt-to-income, etc.)
3. Stratified 80/20 train/test split
4. `RandomizedSearchCV` hyperparameter tuning for Logistic Regression,
   Random Forest, and XGBoost (5-fold CV, scored on ROC-AUC)
5. Held-out evaluation of all 3 tuned models
6. SHAP explainability (global summary + saved background sample)
7. Saves everything to `artifacts/` and `reports/figures/`, and logs
   every run to MLflow (`mlflow.db`)

Expect a few minutes total — Random Forest and XGBoost tuning are the
slow steps.

## 5. Re-check model health without retraining

```bash
python evaluate.py --config config.yaml
```
or
```bash
make evaluate
```

Reloads `artifacts/model_pipeline.joblib`, rebuilds the identical test
split (same `random_state`), and prints a clean metrics report.

## 6. Score a single applicant

```bash
python predict.py --input sample_applicant.json
```
or
```bash
make predict
```

Omit `--input` to score the built-in example applicant. Any JSON file
with the same fields as `sample_applicant.json` works — this is what
you'd point a web form or API endpoint at.

## 7. View experiment tracking

```bash
make mlflow-ui
```
Open `http://localhost:5000` to see every tuning run's hyperparameters
and CV score side by side.

## 8. Push to GitHub

```bash
git init
git add .
git commit -m "Loan Approval Risk Prediction & Explainable AI System"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

`.gitignore` already excludes `mlflow.db`, `mlruns/`, and `__pycache__/`
so your repo stays clean — trained artifacts in `artifacts/` and
`reports/figures/` ARE committed on purpose, since a portfolio reviewer
should be able to see results without running anything.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run every command from the project root, not from inside `src/` |
| MLflow "filestore in maintenance mode" error | Already handled — this repo uses `sqlite:///mlflow.db`, not the raw filesystem backend |
| SHAP explanation shows all-zero values | Make sure `artifacts/shap_background.joblib` exists (created automatically by `train.py`) |
| Want to reset everything | `make clean` removes trained artifacts, figures, and the MLflow DB (raw `data/*.csv` is kept) |
| XGBoost/SHAP install fails on your machine | Both have prebuilt wheels for major platforms via pip; if it still fails, drop `xgboost` from `models.xgboost.enabled: false` in `config.yaml` and remove it from `requirements.txt` |
