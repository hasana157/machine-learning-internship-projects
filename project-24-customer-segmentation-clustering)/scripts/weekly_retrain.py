"""
scripts/weekly_retrain.py
---------------------------
Simplified weekly retraining job: recomputes RFM from the latest
transactions, checks for distribution drift vs. the previous snapshot,
and refits KMeans if drift is detected or the silhouette score would
otherwise degrade. Designed to be triggered by cron / APScheduler / a
CI scheduled workflow.

Run manually:
    python scripts/weekly_retrain.py

Schedule with cron (every Sunday 23:00):
    0 23 * * 0 cd /path/to/CustomerSegmentAI && python scripts/weekly_retrain.py >> logs/retrain.log 2>&1
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.rfm_calculator import RFMCalculator
from src.clustering_engine import ClusteringEngine
from src.persona_generator import PersonaGenerator

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("weekly_retrain")

DRIFT_P_THRESHOLD = 0.05
SILHOUETTE_TOLERANCE = 0.01


def load_previous_rfm() -> pd.DataFrame | None:
    prev_path = DATA_DIR / "rfm_clustered.parquet"
    if not prev_path.exists():
        prev_path = DATA_DIR / "rfm_clustered.csv"
    if not prev_path.exists():
        return None
    return pd.read_parquet(prev_path) if prev_path.suffix == ".parquet" else pd.read_csv(prev_path)


def detect_drift(old_rfm: pd.DataFrame, new_rfm: pd.DataFrame) -> dict:
    results = {}
    for col in ["Recency", "Frequency", "Monetary"]:
        stat, pvalue = ks_2samp(old_rfm[col], new_rfm[col])
        results[col.lower()] = float(pvalue)
    return results


def main():
    LOGS_DIR.mkdir(exist_ok=True)
    logger.info("Starting weekly RFM recomputation")

    try:
        raw_path = DATA_DIR / "raw" / "transactions.csv"
        if not raw_path.exists():
            logger.error("No transaction data at %s -- aborting.", raw_path)
            return

        transactions = pd.read_csv(raw_path, parse_dates=["InvoiceDate"])

        rfm_calc = RFMCalculator()
        rfm_new = rfm_calc.fit_transform(transactions)
        report = rfm_calc.validate(rfm_new)
        if not report.passed:
            logger.warning("RFM validation issues: %s", report.messages)

        old_rfm = load_previous_rfm()
        drift_detected = False
        drift_results = {}
        if old_rfm is not None and len(old_rfm) > 1:
            drift_results = detect_drift(old_rfm, rfm_new)
            drift_detected = any(p < DRIFT_P_THRESHOLD for p in drift_results.values())
            logger.info("Drift test p-values: %s", drift_results)

        engine = ClusteringEngine(k_range=range(3, 6), random_state=42)
        scaled_cols = ["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]
        engine.fit(rfm_new[scaled_cols].to_numpy())
        logger.info(
            "Refit complete: k=%d silhouette=%.3f (drift_detected=%s)",
            engine.best_k, engine.result.silhouette_score, drift_detected,
        )

        rfm_out = rfm_new.copy()
        rfm_out["cluster"] = engine.get_labels()
        rfm_out["silhouette"] = engine.get_silhouette_samples()

        persona_gen = PersonaGenerator()
        persona_gen.fit(rfm_out[["CustomerID", "Recency", "Frequency", "Monetary"]], engine.get_labels())

        # Persist updated artifacts
        rfm_calc.save(MODELS_DIR / "rfm_calculator.pkl")
        engine.save(MODELS_DIR / "clustering_engine.pkl")
        persona_gen.to_json(MODELS_DIR / "personas.json")
        try:
            rfm_out.to_parquet(DATA_DIR / "rfm_clustered.parquet", index=False)
        except (ImportError, ValueError):
            rfm_out.to_csv(DATA_DIR / "rfm_clustered.csv", index=False)

        metrics = {
            "n_customers": int(len(rfm_out)),
            "best_k": int(engine.best_k),
            "silhouette_score": engine.result.silhouette_score,
            "davies_bouldin": engine.result.davies_bouldin,
            "calinski_harabasz": engine.result.calinski_harabasz,
            "inertia": engine.result.inertia,
            "k_search": engine.result.k_search,
            "reference_date": str(rfm_calc.reference_date),
            "drift_detected": drift_detected,
            "drift_pvalues": drift_results,
        }
        with open(MODELS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Weekly segmentation complete. Silhouette: %.3f", engine.result.silhouette_score)

    except Exception:
        logger.exception("Weekly retraining failed")
        raise


if __name__ == "__main__":
    main()
