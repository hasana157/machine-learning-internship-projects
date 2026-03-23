import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def predict_new(model_path, new_data):
    """Load model and make predictions for new data"""
    model = joblib.load(model_path)
    probs = model.predict_proba(new_data)[:,1]
    preds = (probs >= 0.5).astype(int)  # default threshold 0.5
    return preds, probs

if __name__ == "__main__":
    model_path = "models/gb_model.joblib"

    # Path for new data
    new_data_path = Path("data/creditcard_new.csv")

    if new_data_path.exists():
        new_data = pd.read_csv(new_data_path)
        # Remove target column if present
        if 'Class' in new_data.columns:
            new_data = new_data.drop('Class', axis=1)
    else:
        print("creditcard_new.csv not found. Using full dataset instead...")
        new_data = pd.read_csv("data/creditcard.csv").drop('Class', axis=1)

    preds, probs = predict_new(model_path, new_data)
    print("Predictions (0=non-default, 1=default):", preds[:20])  # first 20
    print("Probabilities:", probs[:20])  # first 20
