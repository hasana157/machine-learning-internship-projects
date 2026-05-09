"""
predictor.py
------------
Inference module: loads the trained pipeline and predicts job category
from raw resume text (or structured dict input).

WARNING: Educational/demo only. NOT for real hiring decisions.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.trainer import load_model
from src.data_loader import EDUCATION_MAP
from src.config import TEXT_COLUMN

logger = logging.getLogger(__name__)


def _parse_input(resume_text: str) -> pd.DataFrame:
    """
    Convert a raw text string into a one-row DataFrame matching the
    trained ColumnTransformer's expected input schema.

    We extract education and experience hints from the text if present,
    otherwise default to mid-level values.
    """
    text_lower = resume_text.lower()

    # Heuristic: detect education level from text
    education_ord = 2.0  # default: bachelors
    for edu, val in sorted(EDUCATION_MAP.items(), key=lambda x: -x[1]):
        if edu in text_lower:
            education_ord = float(val)
            break

    # Heuristic: detect years of experience
    import re
    years_exp = 3.0  # default: 3 years
    m = re.search(r"(\d+)\s*(?:years?|yrs?)", text_lower)
    if m:
        years_exp = float(m.group(1))

    # Skill count: count comma-separated tokens in skills-like portion
    skill_count = float(len([t for t in resume_text.split(",") if t.strip()]))
    if skill_count < 1:
        skill_count = float(len(resume_text.split()))

    return pd.DataFrame([{
        TEXT_COLUMN:    resume_text,
        "years_exp":    years_exp,
        "education_ord": education_ord,
        "skill_count":   skill_count,
    }])


class ResumePredictor:
    """
    Wraps the trained sklearn Pipeline for single or batch prediction.

    Usage
    -----
    predictor = ResumePredictor()
    result = predictor.predict("Python, SQL, Machine Learning, 5 years, Masters")
    """

    def __init__(self) -> None:
        self._pipeline = load_model()
        self._classes: List[str] = self._pipeline.classes_.tolist()
        logger.info("ResumePredictor ready. Classes: %s", self._classes)

    def predict(self, resume_text: str) -> Dict:
        """
        Predict job role for a single resume text string.

        Returns dict with:
          predicted_label : str
          confidence      : float (probability of predicted class)
          all_scores      : dict {class_name: probability}
        """
        if not resume_text or not resume_text.strip():
            raise ValueError("resume_text must be a non-empty string.")

        df_input = _parse_input(resume_text)
        proba = self._pipeline.predict_proba(df_input)[0]

        predicted_idx   = int(np.argmax(proba))
        predicted_label = self._classes[predicted_idx]
        confidence      = float(proba[predicted_idx])
        all_scores      = {
            cls: round(float(p), 4)
            for cls, p in zip(self._classes, proba)
        }

        result = {
            "predicted_label": predicted_label,
            "confidence":      round(confidence, 4),
            "all_scores":      all_scores,
        }
        logger.info(
            "Prediction -> %s (confidence=%.1f%%)",
            predicted_label, confidence * 100,
        )
        return result

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        return [self.predict(t) for t in texts]

    @property
    def classes(self) -> List[str]:
        return self._classes


def predict_from_text(text: str) -> Dict:
    """One-shot convenience wrapper."""
    return ResumePredictor().predict(text)
