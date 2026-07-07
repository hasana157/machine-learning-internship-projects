"""
Domain feature engineering for loan risk.

These derived features are the ones an actual credit-risk analyst
would look at before a raw ML model ever gets involved -- income
coverage, debt burden, and employment stability all carry real
signal that raw columns alone under-represent.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["total_income"] = df["applicant_income"] + df["coapplicant_income"]

    # Avoid divide-by-zero; loan_amount is always > 0 by construction/validation
    df["income_to_loan_ratio"] = df["total_income"] / df["loan_amount"].replace(0, np.nan)

    monthly_installment = df["loan_amount"] / df["loan_term_months"]
    df["debt_to_income"] = monthly_installment / (df["total_income"] / 12 + 1)

    # Employment stability: years employed relative to (age - 18), capped at 1
    working_years_possible = (df["age"] - 18).clip(lower=1)
    df["employment_stability"] = (df["employment_years"] / working_years_possible).clip(upper=1)

    # Combines credit score trend with defaults/existing debt into one risk proxy
    df["credit_utilization_risk"] = (
        (850 - df["credit_score"].fillna(df["credit_score"].median())) / 550
        + 0.3 * df["previous_defaults"]
        + 0.15 * df["existing_loans_count"]
    )

    df["income_to_loan_ratio"] = df["income_to_loan_ratio"].replace([np.inf, -np.inf], np.nan)

    return df
