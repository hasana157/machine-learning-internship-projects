import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.persona_generator import PersonaGenerator


@pytest.fixture
def rfm_and_labels():
    rfm = pd.DataFrame({
        "CustomerID": range(1, 13),
        "Recency": [10, 12, 15, 40, 45, 50, 90, 95, 100, 200, 210, 220],
        "Frequency": [80, 90, 85, 30, 25, 28, 10, 12, 9, 2, 1, 3],
        "Monetary": [12000, 13000, 11500, 3000, 2800, 3200, 900, 1000, 850, 100, 90, 120],
    })
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    return rfm, labels


def test_fit_produces_one_persona_per_cluster(rfm_and_labels):
    rfm, labels = rfm_and_labels
    gen = PersonaGenerator()
    gen.fit(rfm, labels)
    personas = gen.generate_personas()
    assert len(personas) == len(set(labels))


def test_persona_names_are_business_meaningful(rfm_and_labels):
    rfm, labels = rfm_and_labels
    gen = PersonaGenerator()
    gen.fit(rfm, labels)
    names = {p["persona_name"] for p in gen.generate_personas()}
    assert names.issubset({"Platinum", "Gold", "Silver", "At-Risk"})


def test_high_value_cluster_is_platinum(rfm_and_labels):
    rfm, labels = rfm_and_labels
    gen = PersonaGenerator()
    gen.fit(rfm, labels)
    personas = {p["id"]: p for p in gen.generate_personas()}
    assert personas[0]["persona_name"] == "Platinum"


def test_lapsed_cluster_is_at_risk(rfm_and_labels):
    rfm, labels = rfm_and_labels
    gen = PersonaGenerator()
    gen.fit(rfm, labels)
    personas = {p["id"]: p for p in gen.generate_personas()}
    assert personas[3]["persona_name"] == "At-Risk"


def test_percentages_sum_to_100(rfm_and_labels):
    rfm, labels = rfm_and_labels
    gen = PersonaGenerator()
    gen.fit(rfm, labels)
    total_pct = sum(p["pct"] for p in gen.generate_personas())
    assert abs(total_pct - 100.0) < 1e-6


def test_generate_before_fit_raises():
    gen = PersonaGenerator()
    with pytest.raises(RuntimeError):
        gen.generate_personas()


def test_to_json_writes_file(tmp_path, rfm_and_labels):
    rfm, labels = rfm_and_labels
    gen = PersonaGenerator()
    gen.fit(rfm, labels)
    path = tmp_path / "personas.json"
    gen.to_json(path)
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert len(data) == len(set(labels))


def test_to_markdown_contains_persona_names(rfm_and_labels):
    rfm, labels = rfm_and_labels
    gen = PersonaGenerator()
    gen.fit(rfm, labels)
    md = gen.to_markdown()
    assert "Platinum" in md
    assert "At-Risk" in md


def test_each_persona_has_actions(rfm_and_labels):
    rfm, labels = rfm_and_labels
    gen = PersonaGenerator()
    gen.fit(rfm, labels)
    for p in gen.generate_personas():
        assert len(p["actions"]) > 0
