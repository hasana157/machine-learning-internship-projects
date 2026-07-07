"""
rfm_calculator.py
------------------
Transforms raw transaction-level data into a per-customer RFM
(Recency, Frequency, Monetary) feature matrix, ready for scaling and
clustering.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class RFMValidationReport:
    n_customers: int
    n_null_values: int
    n_negative_recency: int
    n_negative_frequency: int
    n_negative_monetary: int
    passed: bool
    messages: list


class RFMCalculator:
    """Compute, scale, and validate Recency/Frequency/Monetary features.

    Parameters
    ----------
    rfm_window_days : int
        Only transactions within this many days of the reference date are
        used to compute Frequency and Monetary (Recency always uses the
        single most recent transaction, regardless of window).
    """

    REQUIRED_COLUMNS = ["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID"]

    def __init__(self, rfm_window_days: int = 365):
        self.rfm_window_days = rfm_window_days
        self.scaler: Optional[StandardScaler] = None
        self.reference_date: Optional[pd.Timestamp] = None
        self._feature_cols = ["Recency", "Frequency", "Monetary"]

    # ------------------------------------------------------------------
    # Core RFM computation
    # ------------------------------------------------------------------
    def _clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        clean = df.copy()
        clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"])
        clean = clean.dropna(subset=["CustomerID"])
        clean = clean[clean["Quantity"] > 0]           # drop returns
        clean = clean[clean["UnitPrice"] > 0]
        clean["TotalAmount"] = clean["Quantity"] * clean["UnitPrice"]
        return clean

    def compute_rfm(self, df: pd.DataFrame,
                     reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Compute raw (unscaled) RFM features per customer.

        Returns a DataFrame with columns:
            CustomerID, Recency, Frequency, Monetary
        """
        clean = self._clean_transactions(df)

        ref_date = reference_date or clean["InvoiceDate"].max()
        self.reference_date = ref_date

        window_start = ref_date - pd.Timedelta(days=self.rfm_window_days)
        windowed = clean[clean["InvoiceDate"] >= window_start]

        grouped = windowed.groupby("CustomerID").agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
        ).reset_index()

        grouped["Recency"] = (ref_date - grouped["LastPurchase"]).dt.days

        rfm = grouped[["CustomerID", "Recency", "Frequency", "Monetary"]].copy()
        # Remove inactive / degenerate customers
        rfm = rfm[(rfm["Frequency"] > 0) & (rfm["Monetary"] > 0)]
        rfm = rfm.reset_index(drop=True)
        return rfm

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame,
            reference_date: Optional[pd.Timestamp] = None) -> "RFMCalculator":
        """Compute RFM and fit the StandardScaler on it."""
        rfm = self.compute_rfm(df, reference_date)
        self.scaler = StandardScaler()
        self.scaler.fit(rfm[self._feature_cols])
        return self

    def transform(self, df: pd.DataFrame,
                  reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Compute RFM and return it with scaled columns attached.

        Output columns: CustomerID, Recency, Frequency, Monetary,
                         Recency_scaled, Frequency_scaled, Monetary_scaled
        """
        if self.scaler is None:
            raise RuntimeError("Call fit() before transform().")
        rfm = self.compute_rfm(df, reference_date)
        scaled = self.scaler.transform(rfm[self._feature_cols])
        for i, col in enumerate(self._feature_cols):
            rfm[f"{col}_scaled"] = scaled[:, i]
        return rfm

    def fit_transform(self, df: pd.DataFrame,
                       reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        self.fit(df, reference_date)
        return self.transform(df, reference_date)

    def inverse_transform(self, rfm_scaled: np.ndarray) -> np.ndarray:
        """Convert a (n, 3) scaled RFM array back to original units."""
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fit yet.")
        return self.scaler.inverse_transform(rfm_scaled)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, rfm: pd.DataFrame) -> RFMValidationReport:
        """Run a small set of Great-Expectations-style sanity checks."""
        messages = []
        n_null = int(rfm[self._feature_cols].isnull().sum().sum())
        n_neg_r = int((rfm["Recency"] < 0).sum())
        n_neg_f = int((rfm["Frequency"] <= 0).sum())
        n_neg_m = int((rfm["Monetary"] <= 0).sum())

        if n_null:
            messages.append(f"Found {n_null} null values in R/F/M columns.")
        if n_neg_r:
            messages.append(f"Found {n_neg_r} rows with negative Recency.")
        if n_neg_f:
            messages.append(f"Found {n_neg_f} rows with non-positive Frequency.")
        if n_neg_m:
            messages.append(f"Found {n_neg_m} rows with non-positive Monetary.")

        passed = not messages
        if passed:
            messages.append("All RFM validation checks passed.")

        return RFMValidationReport(
            n_customers=len(rfm),
            n_null_values=n_null,
            n_negative_recency=n_neg_r,
            n_negative_frequency=n_neg_f,
            n_negative_monetary=n_neg_m,
            passed=passed,
            messages=messages,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "RFMCalculator":
        with open(path, "rb") as f:
            return pickle.load(f)
