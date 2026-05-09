"""
data_loader.py
--------------
Loads the CSV dataset, performs feature engineering, and validates label integrity.

NOTE: This dataset is *synthetic / curated* with identical skill vocabulary across
      all roles. DO NOT use for real hiring decisions. See README.
"""

import pandas as pd
import logging
from typing import Tuple

from src.config import (
    DATASET_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
    ALLOWED_LABELS,
)

logger = logging.getLogger(__name__)

# Education ordinal mapping
EDUCATION_MAP = {
    "high school": 0,
    "associate":   1,
    "bachelors":   2,
    "masters":     3,
    "phd":         4,
}


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load raw CSV -> clean -> engineer feature columns.

    Feature Engineering
    -------------------
    Since all roles share the exact same 10 skills in this synthetic dataset,
    we use a multi-feature strategy:

    1. resume_text  : skills string (for TF-IDF; captures skill combinations)
    2. years_exp    : numeric years (kept as integer for numeric sub-pipeline)
    3. education_ord: ordinal-encoded education level (0=HighSchool..4=PhD)
    4. skill_count  : number of skills listed (signal for seniority/breadth)

    Returns a DataFrame ready for the ColumnTransformer pipeline.
    """
    logger.info("Loading dataset from: %s", path)
    df = pd.read_csv(path)

    _validate_columns(df)
    df = _clean(df)
    df = _engineer_features(df)
    df = _filter_labels(df)

    logger.info(
        "Dataset loaded -- %d samples | %d classes",
        len(df),
        df[LABEL_COLUMN].nunique(),
    )
    _log_distribution(df)
    return df


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature DataFrame and label Series."""
    feature_cols = [TEXT_COLUMN, "years_exp", "education_ord", "skill_count"]
    return df[feature_cols], df[LABEL_COLUMN]


# Private helpers

def _validate_columns(df: pd.DataFrame) -> None:
    required = {"Name", "YearsExperience", "Skills", "Education", "JobRole"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(subset=["Skills", "Education", "JobRole"], inplace=True)
    df["YearsExperience"] = pd.to_numeric(
        df["YearsExperience"], errors="coerce"
    ).fillna(0).astype(int)
    df["Skills"]    = df["Skills"].str.strip()
    df["Education"] = df["Education"].str.strip()
    df["JobRole"]   = df["JobRole"].str.strip()
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Text feature: normalised skills (space-separated, lowercase)
    df[TEXT_COLUMN] = (
        df["Skills"]
        .str.lower()
        .str.replace(",", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 2. Numeric: years of experience
    df["years_exp"] = df["YearsExperience"].astype(float)

    # 3. Ordinal: education level
    df["education_ord"] = (
        df["Education"]
        .str.lower()
        .str.strip()
        .map(EDUCATION_MAP)
        .fillna(2)  # default to bachelors if unknown
        .astype(float)
    )

    # 4. Skill count (breadth signal)
    df["skill_count"] = df["Skills"].str.split(",").apply(len).astype(float)

    return df


def _filter_labels(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df[LABEL_COLUMN].isin(ALLOWED_LABELS)].copy()
    removed = before - len(df)
    if removed:
        logger.warning("Removed %d rows with unsupported labels.", removed)
    return df


def _log_distribution(df: pd.DataFrame) -> None:
    dist = df[LABEL_COLUMN].value_counts()
    logger.info("Class distribution:\n%s", dist.to_string())
