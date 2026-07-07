"""
Data ingestion & validation.

Loads data/loan_applications.csv if present; otherwise generates a
realistic synthetic loan-applicant dataset (so the pipeline runs
end-to-end with zero downloads). Every load goes through
`validate_schema` -- the same guard rail you'd want before feeding
anything into a banking risk model.

Swap in a real dataset (e.g. Kaggle "Loan Prediction Problem Dataset")
by dropping it at `data/loan_applications.csv` with matching columns --
see README "Using a Real Dataset".
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "applicant_income", "coapplicant_income", "loan_amount", "loan_term_months",
    "employment_years", "age", "dependents", "previous_defaults",
    "existing_loans_count", "credit_score", "education", "self_employed",
    "property_area", "marital_status", "loan_approved",
]


def generate_synthetic_data(config: dict) -> pd.DataFrame:
    cfg = config["data"]["synthetic"]
    n = config["data"]["n_samples"]
    rng = np.random.default_rng(cfg["seed"])

    applicant_income = rng.gamma(shape=5, scale=1200, size=n) + 800
    coapplicant_income = rng.choice([0, 1], size=n, p=[0.35, 0.65]) * \
        (rng.gamma(shape=4, scale=800, size=n))
    loan_amount = rng.gamma(shape=6, scale=25000 / 6, size=n) + 5000
    loan_term_months = rng.choice([120, 180, 240, 300, 360], size=n,
                                   p=[0.1, 0.15, 0.2, 0.25, 0.3])
    employment_years = np.clip(rng.normal(7, 5, n), 0, 40)
    age = np.clip(rng.normal(38, 10, n), 21, 70).round()
    dependents = rng.choice([0, 1, 2, 3], size=n, p=[0.45, 0.25, 0.2, 0.1])
    previous_defaults = rng.choice([0, 1, 2], size=n, p=[0.78, 0.17, 0.05])
    existing_loans_count = rng.choice([0, 1, 2, 3], size=n, p=[0.5, 0.3, 0.15, 0.05])
    credit_score = np.clip(rng.normal(650, 80, n), 300, 850).round()
    education = rng.choice(["Graduate", "Not Graduate"], size=n, p=[0.7, 0.3])
    self_employed = rng.choice(["Yes", "No"], size=n, p=[0.2, 0.8])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], size=n, p=[0.4, 0.35, 0.25])
    marital_status = rng.choice(["Married", "Single"], size=n, p=[0.6, 0.4])

    total_income = applicant_income + coapplicant_income
    monthly_installment = loan_amount / loan_term_months
    dti = monthly_installment / (total_income + 1)

    # Latent "risk score" driving approval -- built from realistic banking
    # signals so the target is learnable but not trivially separable.
    risk_logit = (
        -0.00012 * total_income
        + 7.0 * dti
        - 0.009 * (credit_score - 650)
        + 0.9 * previous_defaults
        + 0.35 * existing_loans_count
        - 0.05 * employment_years
        + 0.15 * dependents
        + rng.normal(0, 0.35, n)
    )
    approval_prob = 1 / (1 + np.exp(risk_logit - 1.2))
    loan_approved = (rng.uniform(0, 1, n) < approval_prob).astype(int)

    df = pd.DataFrame({
        "applicant_income": applicant_income.round(2),
        "coapplicant_income": coapplicant_income.round(2),
        "loan_amount": loan_amount.round(2),
        "loan_term_months": loan_term_months,
        "employment_years": employment_years.round(1),
        "age": age,
        "dependents": dependents,
        "previous_defaults": previous_defaults,
        "existing_loans_count": existing_loans_count,
        "credit_score": credit_score,
        "education": education,
        "self_employed": self_employed,
        "property_area": property_area,
        "marital_status": marital_status,
        "loan_approved": loan_approved,
    })

    # Sprinkle a few missing values -- realistic banking data is never clean,
    # and it exercises the imputers in the preprocessing pipeline.
    for col in ["credit_score", "employment_years", "self_employed"]:
        mask = rng.uniform(0, 1, n) < 0.03
        df.loc[mask, col] = np.nan

    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    if df["loan_approved"].dropna().isin([0, 1]).all() is False:
        raise ValueError("Target column 'loan_approved' must be binary (0/1).")

    null_frac = df.isnull().mean()
    high_null_cols = null_frac[null_frac > 0.5].index.tolist()
    if high_null_cols:
        raise ValueError(f"Columns with >50% missing values, check data source: {high_null_cols}")

    if len(df) < 100:
        raise ValueError(f"Dataset too small for reliable training: {len(df)} rows.")

    logger.info(f"Schema validation passed: {len(df)} rows, {len(df.columns)} columns, "
                f"{df.isnull().sum().sum()} missing cells.")


def load_data(config: dict) -> pd.DataFrame:
    raw_path = Path(config["data"]["raw_path"])

    if raw_path.exists():
        logger.info(f"Loading existing dataset -> {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        logger.info("No dataset found -- generating synthetic loan applications...")
        df = generate_synthetic_data(config)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        logger.info(f"Synthetic dataset saved -> {raw_path}")

    validate_schema(df)
    return df
