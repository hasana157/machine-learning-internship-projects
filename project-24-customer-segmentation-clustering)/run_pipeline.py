"""
run_pipeline.py
-----------------
End-to-end batch pipeline:
    1. Load transaction data (generates synthetic data if none exists)
    2. Compute + scale RFM features (RFMCalculator)
    3. Fit KMeans with silhouette-based k selection (ClusteringEngine)
    4. Generate business personas (PersonaGenerator)
    5. Save all artifacts to models/ so app.py can load them instantly

Run:
    python run_pipeline.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.rfm_calculator import RFMCalculator
from src.clustering_engine import ClusteringEngine
from src.persona_generator import PersonaGenerator

ROOT = Path(__file__).parent
RAW_PATH = ROOT / "data" / "raw" / "transactions.csv"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


def load_or_generate_transactions() -> pd.DataFrame:
    if not RAW_PATH.exists():
        print(f"No transaction data found at {RAW_PATH}. Generating synthetic dataset...")
        from data.generate_synthetic_data import generate
        df = generate(n_customers=5000, n_transactions=50000)
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_PATH, index=False)
    else:
        df = pd.read_csv(RAW_PATH, parse_dates=["InvoiceDate"])
    return df


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1/4 -- Loading transaction data")
    print("=" * 60)
    transactions = load_or_generate_transactions()
    print(f"Loaded {len(transactions):,} transactions, "
          f"{transactions['CustomerID'].nunique():,} unique customers")

    print("\n" + "=" * 60)
    print("STEP 2/4 -- RFM feature engineering")
    print("=" * 60)
    rfm_calc = RFMCalculator(rfm_window_days=365)
    rfm = rfm_calc.fit_transform(transactions)
    report = rfm_calc.validate(rfm)
    for msg in report.messages:
        print(f"  - {msg}")
    if not report.passed:
        warnings.warn("RFM validation reported issues; review before production use.")
    print(f"RFM matrix shape: {rfm.shape}")

    rfm.to_parquet(DATA_DIR / "rfm.parquet", index=False) if _has_parquet() else \
        rfm.to_csv(DATA_DIR / "rfm.csv", index=False)
    rfm_calc.save(MODELS_DIR / "rfm_calculator.pkl")

    scaled_cols = ["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]
    rfm_scaled = rfm[scaled_cols].to_numpy()

    print("\n" + "=" * 60)
    print("STEP 3/4 -- KMeans clustering + silhouette optimization")
    print("=" * 60)
    # k restricted to 3-5 per the business persona-tier requirement
    # (Platinum / Gold / Silver / At-Risk); widen to range(2, 11) for a
    # pure statistical exploration of k.
    engine = ClusteringEngine(k_range=range(3, 6), random_state=42)
    engine.fit(rfm_scaled)
    result = engine.result
    print(f"Optimal k: {result.best_k}")
    print(f"Silhouette Score: {result.silhouette_score:.3f}")
    print(f"Davies-Bouldin Index: {result.davies_bouldin:.3f} (lower is better)")
    print(f"Calinski-Harabasz Index: {result.calinski_harabasz:.1f} (higher is better)")
    for row in result.k_search:
        print(f"    k={row['k']:>2}  silhouette={row['silhouette']:.3f}  inertia={row['inertia']:.1f}")

    engine.save(MODELS_DIR / "clustering_engine.pkl")
    labels = engine.get_labels()
    sil_samples = engine.get_silhouette_samples()

    rfm_out = rfm.copy()
    rfm_out["cluster"] = labels
    rfm_out["silhouette"] = sil_samples
    if _has_parquet():
        rfm_out.to_parquet(DATA_DIR / "rfm_clustered.parquet", index=False)
    else:
        rfm_out.to_csv(DATA_DIR / "rfm_clustered.csv", index=False)
    np.save(MODELS_DIR / "silhouette_samples.npy", sil_samples)

    print("\n" + "=" * 60)
    print("STEP 4/4 -- Persona generation")
    print("=" * 60)
    persona_gen = PersonaGenerator()
    persona_gen.fit(rfm_out[["CustomerID", "Recency", "Frequency", "Monetary"]], labels)
    personas = persona_gen.generate_personas()
    for p in personas:
        print(f"  Cluster {p['id']}: {p['persona_name']} "
              f"({p['count']} customers, {p['pct']:.1f}%) "
              f"-- R={p['r_mean']:.0f}d F={p['f_mean']:.1f} M={p['m_mean']:.0f}")

    persona_gen.to_json(MODELS_DIR / "personas.json")
    with open(MODELS_DIR / "personas.md", "w") as f:
        f.write(persona_gen.to_markdown())

    # Save headline metrics for the dashboard
    import json
    metrics = {
        "n_customers": int(len(rfm_out)),
        "best_k": int(result.best_k),
        "silhouette_score": result.silhouette_score,
        "davies_bouldin": result.davies_bouldin,
        "calinski_harabasz": result.calinski_harabasz,
        "inertia": result.inertia,
        "k_search": result.k_search,
        "reference_date": str(rfm_calc.reference_date),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nPipeline complete. Artifacts saved to:", MODELS_DIR)
    print("Run the dashboard with:  streamlit run app.py")


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


if __name__ == "__main__":
    main()
