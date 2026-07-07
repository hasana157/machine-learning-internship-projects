"""
api/main.py -- Optional FastAPI real-time scoring service
------------------------------------------------------------
Exposes POST /predict-cluster: given a customer's raw R/F/M values,
returns the assigned cluster, persona name, and recommended action.

Run:
    uvicorn api.main:app --reload --port 8000

Prerequisite: run `python run_pipeline.py` first so models/*.pkl and
models/personas.json exist.
"""
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.rfm_calculator import RFMCalculator
from src.clustering_engine import ClusteringEngine

app = FastAPI(
    title="CustomerSegment AI Scoring API",
    description="Real-time RFM -> cluster -> persona assignment",
    version="1.0.0",
)

_rfm_calculator: RFMCalculator | None = None
_clustering_engine: ClusteringEngine | None = None
_personas: list | None = None


class RFMInput(BaseModel):
    recency: float = Field(..., ge=0, description="Days since last purchase")
    frequency: float = Field(..., gt=0, description="Number of purchases in the RFM window")
    monetary: float = Field(..., gt=0, description="Total spend in the RFM window")


class ClusterPrediction(BaseModel):
    cluster: int
    persona: str
    recommendation: list


@app.on_event("startup")
def load_artifacts():
    global _rfm_calculator, _clustering_engine, _personas
    models_dir = ROOT / "models"
    try:
        _rfm_calculator = RFMCalculator.load(models_dir / "rfm_calculator.pkl")
        _clustering_engine = ClusteringEngine.load(models_dir / "clustering_engine.pkl")
        with open(models_dir / "personas.json") as f:
            _personas = json.load(f)
    except FileNotFoundError:
        # Artifacts not built yet; endpoints will return a clear 503 until
        # `python run_pipeline.py` has been run.
        _rfm_calculator = None
        _clustering_engine = None
        _personas = None


@app.get("/health")
def health():
    ready = all([_rfm_calculator, _clustering_engine, _personas])
    return {"status": "ready" if ready else "not_ready"}


@app.post("/predict-cluster", response_model=ClusterPrediction)
def predict_cluster(rfm: RFMInput):
    if not all([_rfm_calculator, _clustering_engine, _personas]):
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run `python run_pipeline.py` first.",
        )

    scaled = _rfm_calculator.scaler.transform([[rfm.recency, rfm.frequency, rfm.monetary]])
    cluster = int(_clustering_engine.predict(scaled)[0])

    persona = next((p for p in _personas if p["id"] == cluster), None)
    if persona is None:
        raise HTTPException(status_code=500, detail=f"No persona found for cluster {cluster}")

    return ClusterPrediction(
        cluster=cluster,
        persona=persona["persona_name"],
        recommendation=persona["actions"],
    )
