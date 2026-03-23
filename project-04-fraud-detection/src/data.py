# src/data.py
from pathlib import Path
import pandas as pd

def load_dataset():
    """
    Load credit card fraud dataset
    Returns:
        X: features DataFrame
        y: target Series
    """

    # Compute the path relative to this file
    data_path = Path(__file__).parent.parent / "data" / "raw" / "creditcard.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find dataset at {data_path.resolve()}")

    df = pd.read_csv(data_path)

    # Drop 'Time' column if present
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    X = df.drop(columns=["Class"])
    y = df["Class"]

    return X, y