import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rfm_calculator import RFMCalculator


@pytest.fixture
def sample_transactions():
    rng = np.random.default_rng(0)
    n = 500
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    df = pd.DataFrame({
        "InvoiceNo": [str(1000 + i) for i in range(n)],
        "StockCode": "A1",
        "Description": "Widget",
        "Quantity": rng.integers(1, 10, n),
        "InvoiceDate": dates,
        "UnitPrice": rng.uniform(1, 50, n),
        "CustomerID": rng.integers(1, 50, n),
        "Country": "United Kingdom",
    })
    return df


def test_compute_rfm_shape_and_columns(sample_transactions):
    calc = RFMCalculator()
    rfm = calc.compute_rfm(sample_transactions)
    assert set(["CustomerID", "Recency", "Frequency", "Monetary"]).issubset(rfm.columns)
    assert len(rfm) <= sample_transactions["CustomerID"].nunique()


def test_rfm_values_are_positive(sample_transactions):
    calc = RFMCalculator()
    rfm = calc.compute_rfm(sample_transactions)
    assert (rfm["Recency"] >= 0).all()
    assert (rfm["Frequency"] > 0).all()
    assert (rfm["Monetary"] > 0).all()


def test_returns_are_excluded():
    df = pd.DataFrame({
        "InvoiceNo": ["1", "2"],
        "InvoiceDate": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "Quantity": [5, -3],
        "UnitPrice": [10.0, 10.0],
        "CustomerID": [1, 1],
    })
    calc = RFMCalculator()
    rfm = calc.compute_rfm(df)
    assert len(rfm) == 1
    assert rfm.iloc[0]["Frequency"] == 1


def test_missing_column_raises():
    df = pd.DataFrame({"InvoiceDate": [], "Quantity": [], "UnitPrice": []})
    calc = RFMCalculator()
    with pytest.raises(ValueError):
        calc.compute_rfm(df)


def test_fit_transform_produces_scaled_columns(sample_transactions):
    calc = RFMCalculator()
    rfm = calc.fit_transform(sample_transactions)
    for col in ["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]:
        assert col in rfm.columns
    # Scaled features should have ~zero mean
    assert abs(rfm["Recency_scaled"].mean()) < 1e-6


def test_transform_without_fit_raises(sample_transactions):
    calc = RFMCalculator()
    with pytest.raises(RuntimeError):
        calc.transform(sample_transactions)


def test_inverse_transform_roundtrip(sample_transactions):
    calc = RFMCalculator()
    rfm = calc.fit_transform(sample_transactions)
    scaled = rfm[["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]].to_numpy()
    original = calc.inverse_transform(scaled)
    np.testing.assert_allclose(
        original, rfm[["Recency", "Frequency", "Monetary"]].to_numpy(), atol=1e-6
    )


def test_validate_passes_on_clean_data(sample_transactions):
    calc = RFMCalculator()
    rfm = calc.fit_transform(sample_transactions)
    report = calc.validate(rfm)
    assert report.passed
    assert report.n_null_values == 0


def test_validate_flags_negative_values(sample_transactions):
    calc = RFMCalculator()
    rfm = calc.fit_transform(sample_transactions)
    rfm.loc[0, "Recency"] = -5
    report = calc.validate(rfm)
    assert not report.passed
    assert report.n_negative_recency == 1


def test_save_and_load_roundtrip(tmp_path, sample_transactions):
    calc = RFMCalculator()
    calc.fit(sample_transactions)
    save_path = tmp_path / "calc.pkl"
    calc.save(save_path)
    loaded = RFMCalculator.load(save_path)
    assert loaded.scaler is not None
    rfm1 = calc.transform(sample_transactions)
    rfm2 = loaded.transform(sample_transactions)
    pd.testing.assert_frame_equal(rfm1, rfm2)


def test_reference_date_override(sample_transactions):
    calc = RFMCalculator()
    ref = pd.Timestamp("2025-01-01")
    rfm = calc.compute_rfm(sample_transactions, reference_date=ref)
    assert calc.reference_date == ref


def test_window_filters_old_transactions():
    df = pd.DataFrame({
        "InvoiceNo": ["1", "2"],
        "InvoiceDate": pd.to_datetime(["2020-01-01", "2024-01-01"]),
        "Quantity": [1, 1],
        "UnitPrice": [10.0, 10.0],
        "CustomerID": [1, 1],
    })
    calc = RFMCalculator(rfm_window_days=30)
    rfm = calc.compute_rfm(df, reference_date=pd.Timestamp("2024-01-02"))
    assert rfm.iloc[0]["Frequency"] == 1


def test_empty_dataframe_after_cleaning_returns_empty():
    df = pd.DataFrame({
        "InvoiceNo": ["1"],
        "InvoiceDate": pd.to_datetime(["2024-01-01"]),
        "Quantity": [-1],
        "UnitPrice": [10.0],
        "CustomerID": [1],
    })
    calc = RFMCalculator()
    rfm = calc.compute_rfm(df)
    assert len(rfm) == 0
