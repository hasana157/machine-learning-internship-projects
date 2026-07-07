"""
Single-applicant prediction CLI with explanation.

Usage:
    python predict.py --input sample_applicant.json
    python predict.py                      # uses a built-in example applicant

Outputs the approval decision, the risk probability, and the top
features (with SHAP values) driving that specific decision -- the
"explainable" part of the system, not just a bare label.
"""

import argparse
import json

import joblib
import pandas as pd

from src.explainer import explain_single_prediction
from src.features import engineer_features
from src.preprocessing import get_feature_names
from src.utils import get_logger, load_config

logger = get_logger("predict")

EXAMPLE_APPLICANT = {
    "applicant_income": 4200,
    "coapplicant_income": 1800,
    "loan_amount": 120000,
    "loan_term_months": 360,
    "employment_years": 3.5,
    "age": 29,
    "dependents": 1,
    "previous_defaults": 0,
    "existing_loans_count": 1,
    "credit_score": 610,
    "education": "Graduate",
    "self_employed": "No",
    "property_area": "Semiurban",
    "marital_status": "Married",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Score a single loan applicant.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input", type=str, default=None,
                         help="Path to a JSON file with one applicant's fields. "
                              "Omit to use the built-in example applicant.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.input:
        with open(args.input) as f:
            applicant = json.load(f)
    else:
        applicant = EXAMPLE_APPLICANT
        logger.info("No --input provided, scoring the built-in example applicant.")

    df_row = pd.DataFrame([applicant])
    df_row = engineer_features(df_row)

    pipeline = joblib.load(config["paths"]["best_pipeline"])
    proba = float(pipeline.predict_proba(df_row)[0, 1])
    decision = "APPROVE" if proba >= 0.5 else "REJECT"

    feature_names = get_feature_names(pipeline.named_steps["preprocessor"])
    top_factors = explain_single_prediction(pipeline, df_row, feature_names, top_n=5)

    print("\n" + "=" * 60)
    print("LOAN APPLICATION DECISION")
    print("=" * 60)
    print(f"Decision            : {decision}")
    print(f"Approval probability: {proba:.1%}")
    print(f"Risk probability    : {1 - proba:.1%}")

    if top_factors:
        print("\nTop factors behind this decision (SHAP):")
        for name, value in top_factors:
            clean_name = name.split("__", 1)[-1]
            direction = "→ increases approval odds" if value > 0 else "→ decreases approval odds"
            print(f"  {clean_name:30s} {value:+.3f}  {direction}")
    else:
        print("\n(SHAP explanation unavailable for this model/environment.)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
