# CreditLens — Loan Approval Risk Prediction & Explainable AI System

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-8A2BE2.svg)](https://shap.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

### *"Not just approve or reject — show your work."*

**CreditLens** is a config-driven credit risk pipeline that predicts loan
approval, outputs a calibrated risk probability, and explains **why** —
per applicant — using SHAP. Built to mirror how a real banking ML team
would structure this, not a notebook demo.

</div>

---

## 📌 Overview

Loan approval is a business decision, not just a classification label. A
bank needs three things from a model like this: a reliable **decision**, a
trustworthy **probability** (calibration matters as much as accuracy), and
an **explanation** it can put in front of a regulator or a rejected
applicant. CreditLens is built around all three.

- **Reproducible CLI training** — `python train.py --config config.yaml`
- **Multi-model comparison with tuning** — Logistic Regression, Random Forest, XGBoost, each hyperparameter-searched
- **Business metrics, not just accuracy** — ROC-AUC, PR-AUC, F1, and a calibration curve
- **Per-applicant explanations** — SHAP values behind every individual decision
- **Experiment tracking** — every tuning run logged to MLflow
- **Fully configurable** via `config.yaml` — features, models, and hyperparameter grids, zero hardcoding

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🏗️ **Modular Pipeline** | `data_ingestion.py`, `preprocessing.py`, `model_factory.py`, `trainer.py`, `evaluator.py`, `explainer.py` |
| 🔧 **Domain Feature Engineering** | income-to-loan ratio, debt-to-income, employment stability, credit utilization risk |
| ⚙️ **ColumnTransformer Preprocessing** | leakage-safe imputation + scaling + one-hot, fit only inside CV folds |
| 🤖 **3 Tuned Candidate Models** | Logistic Regression, Random Forest, XGBoost — each via `RandomizedSearchCV` |
| 📊 **Business-Focused Evaluation** | ROC-AUC, PR-AUC, F1, Brier score, calibration curve, confusion matrix |
| 🔍 **Explainable AI** | Global SHAP summary + per-prediction top-5 contributing factors |
| 📈 **MLflow Tracking** | every candidate's params + CV score logged automatically |
| 🖥️ **Full CLI** | `train.py`, `evaluate.py`, `predict.py` — no notebook required to run any of it |

---

## 🧠 How It Works

```text
╔══════════════╗   ╔═══════════════════╗   ╔═══════════════════╗   ╔══════════════════════╗   ╔══════════════════╗
║ Data Ingest  ║──▶║ Feature           ║──▶║ ColumnTransformer  ║──▶║ 3 Models × Randomized ║──▶║ Best Model +      ║
║ + Validation ║   ║ Engineering       ║   ║ Preprocessing      ║   ║ Search (5-fold CV)    ║   ║ SHAP Explanations ║
╚══════════════╝   ╚═══════════════════╝   ╚═══════════════════╝   ╚══════════════════════╝   ╚══════════════════╝
 Schema checks       income-to-loan            impute + scale          LogReg / RF / XGB          ROC/PR/calibration
 Synthetic or         debt-to-income            one-hot categorical     scored on ROC-AUC           SHAP summary +
 real CSV             employment stability                              logged to MLflow            per-row factors
```

### Why these particular metrics?

- **ROC-AUC** — ranks applicants by risk regardless of the approval threshold a bank ultimately chooses.
- **PR-AUC** — with an imbalanced target (most applicants get approved), precision/recall on the minority class matters more than overall accuracy.
- **Calibration curve / Brier score** — a "73% risk" prediction should actually default ~73% of the time. Banks price risk off the *probability*, not just the label, so calibration is a first-class metric here, not an afterthought.

---

## 📊 Results (bundled synthetic dataset, 1,200-row held-out test set)

| Model | ROC-AUC | PR-AUC | F1 | Brier Score |
|---|---|---|---|---|
| **Logistic Regression (selected)** | **0.768** | **0.925** | 0.791 | 0.199 |
| Random Forest | 0.755 | 0.920 | 0.796 | — |
| XGBoost | 0.758 | 0.920 | **0.904** | — |

> Model selection is driven by cross-validated ROC-AUC (see `config.yaml → training.scoring`).
> Swap it to `f1` or `average_precision` if your business priority differs — no code changes needed.
> Results regenerate every time you run `make train`; numbers here are from the bundled synthetic dataset.

**Top global risk drivers (SHAP):** debt-to-income ratio and the engineered
credit-utilization-risk feature dominate, followed by credit score and
previous defaults — consistent with real underwriting practice.

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/hasana157/CreditLens && cd CreditLens

# 2. Install dependencies
make setup

# 3. Train all 3 models with hyperparameter tuning (reproducible, config-driven)
make train

# 4. Re-check model health any time without retraining
make evaluate

# 5. Score a single applicant with a full explanation
make predict
```

`make train` takes a few minutes (hyperparameter search across 3 models).
Every step is logged to the console and to MLflow.

---

## 🗄️ Dataset

### Default: synthetic data (works out of the box)
`train.py` auto-generates `data/loan_applications.csv` on first run if it
doesn't exist — 6,000 realistic synthetic applicants with income, credit
score, employment history, defaults, and a target built from a genuine
risk formula (not random labels), including a small amount of injected
missingness so the imputers actually get exercised.

### Using a real dataset
CreditLens works with any loan-applicant CSV that has these columns
(rename yours to match, or edit `columns:` in `config.yaml`):

```
applicant_income, coapplicant_income, loan_amount, loan_term_months,
employment_years, age, dependents, previous_defaults,
existing_loans_count, credit_score, education, self_employed,
property_area, marital_status, loan_approved
```

Good free sources:
- Kaggle — [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- Kaggle — [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

**Setup steps:**
1. Download and rename/remap columns to match the schema above.
2. Save the CSV as `data/loan_applications.csv`, replacing the synthetic file.
3. Run `make train` — the schema validator in `data_ingestion.py` will
   immediately tell you if a required column is missing or the target
   isn't binary, before any model gets near it.

---

## 📁 Project Structure

```
CreditLens/
│
├── data/
│   └── loan_applications.csv     # Default synthetic dataset (replaceable)
│
├── artifacts/
│   ├── model_pipeline.joblib     # Full preprocessing + model, ready to predict()
│   ├── metrics.json              # ROC-AUC/PR-AUC/F1/Brier + confusion matrix, all models
│   ├── training_summary.json     # CV scores + best hyperparameters, all models
│   ├── feature_importance.csv    # Ranked feature importances (best model)
│   └── shap_background.joblib    # Reference distribution used for per-row SHAP explanations
│
├── reports/figures/
│   ├── roc_curve.png             # All 3 models overlaid
│   ├── precision_recall_curve.png
│   ├── calibration_curve.png
│   ├── feature_importance.png
│   └── shap_summary.png          # Global explainability chart
│
├── src/
│   ├── data_ingestion.py         # Load/generate + schema validation
│   ├── features.py               # Domain feature engineering
│   ├── preprocessing.py          # ColumnTransformer (impute + scale + one-hot)
│   ├── model_factory.py          # Candidate models + hyperparameter grids
│   ├── trainer.py                # RandomizedSearchCV + MLflow logging + model selection
│   ├── evaluator.py               # Business metrics + all report plots
│   ├── explainer.py               # SHAP: global summary + per-prediction factors
│   └── utils.py                   # Config loader + logger
│
├── config.yaml                    # Every path, feature list, model, and hyperparameter grid
├── train.py                       # CLI: python train.py --config config.yaml
├── evaluate.py                    # CLI: reload saved pipeline, recheck metrics
├── predict.py                     # CLI: score one applicant + explain the decision
├── sample_applicant.json          # Example input for predict.py
├── Makefile                       # setup / train / evaluate / predict / mlflow-ui / clean
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration (`config.yaml`)

Nothing is hardcoded in source — every knob lives here:

```yaml
data:
  n_samples: 6000
  test_size: 0.20
  target_column: "loan_approved"

training:
  cv_folds: 5
  scoring: "roc_auc"       # switch to "f1" or "average_precision" if priorities differ
  n_iter_search: 15

models:
  logistic_regression: { enabled: true, param_grid: {...} }
  random_forest:        { enabled: true, param_grid: {...} }
  xgboost:              { enabled: true, param_grid: {...} }

mlflow:
  enabled: true
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "loan-approval-risk"
```

Disable a model entirely by setting `enabled: false` — no code changes
needed. Add a hyperparameter to any grid and `RandomizedSearchCV` picks
it up automatically.

---

## 🔍 Explainability in Practice

```bash
python predict.py --input sample_applicant.json
```

```
============================================================
LOAN APPLICATION DECISION
============================================================
Decision            : REJECT
Approval probability: 5.1%
Risk probability    : 94.9%

Top factors behind this decision (SHAP):
  debt_to_income                 -0.909  → decreases approval odds
  credit_utilization_risk        -0.825  → decreases approval odds
  credit_score                   -0.423  → decreases approval odds
  previous_defaults              -0.238  → decreases approval odds
  employment_years               -0.226  → decreases approval odds
============================================================
```

This is the difference between a model that says "no" and a system a loan
officer (or a regulator) can actually act on.

---

## 📈 Experiment Tracking

Every `make train` run logs each candidate model's hyperparameters and
cross-validated score to MLflow:

```bash
make mlflow-ui
# open http://localhost:5000
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Machine Learning | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Data Processing | pandas, numpy |
| Visualization | Matplotlib |
| Configuration | PyYAML |

---

## 🗺️ Roadmap

- [ ] LightGBM as a fourth candidate model
- [ ] Threshold optimization tool (business-cost-weighted, not just 0.5)
- [ ] FastAPI serving layer around `artifacts/model_pipeline.joblib`
- [ ] Streamlit reviewer dashboard (batch scoring + SHAP waterfall per row)
- [ ] Fairness/bias audit across protected attributes

---

## 👩‍💻 Author

**Hasana Zahid**
AI & ML Engineer | Python Developer

[![GitHub](https://img.shields.io/badge/GitHub-hasana157-black?logo=github)](https://github.com/hasana157)

---

<div align="center">

⭐ **If CreditLens helped you, consider starring the repo!** ⭐

</div>
