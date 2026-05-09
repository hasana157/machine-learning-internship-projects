"""
trainer.py
----------
Constructs a scikit-learn Pipeline using ColumnTransformer:

    text column  -> TextCleaner -> TF-IDF
    numeric cols -> StandardScaler
    -> concatenate -> Logistic Regression

This multi-feature approach extracts signal from both the skills text
AND the numeric/ordinal features (years experience, education, skill count).
"""

import logging
from typing import Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    LR_C,
    LR_MAX_ITER,
    LR_SOLVER,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMN,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    MODEL_DIR,
    LABEL_MAP_PATH,
)
from src.preprocessor import clean_text
from src.utils import ensure_dirs, save_json

logger = logging.getLogger(__name__)

# Numeric feature columns (passed to StandardScaler)
NUMERIC_COLS = ["years_exp", "education_ord", "skill_count"]


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Picklable sklearn transformer that applies clean_text to each document.
    Must be a named class (not a lambda) for joblib serialisation.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [clean_text(doc) for doc in X]


def build_pipeline() -> Pipeline:
    """
    Return an unfitted sklearn Pipeline.

    Architecture
    ------------
    ColumnTransformer:
      - text branch  : TextCleaner -> TfidfVectorizer
      - numeric branch: StandardScaler (years_exp, education_ord, skill_count)
    Final estimator: LogisticRegression (balanced class weights)
    """
    text_pipeline = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf",   TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=True,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("text",    text_pipeline,  TEXT_COLUMN),
            ("numeric", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        C=LR_C,
        max_iter=LR_MAX_ITER,
        solver=LR_SOLVER,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])
    logger.info("Pipeline built (TF-IDF + numeric features + LogReg)")
    return pipeline


def split_data(X, y) -> Tuple:
    """Stratified train/validation split."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info("Split -> train: %d | val: %d", len(X_train), len(X_val))
    return X_train, X_val, y_train, y_val


def train(X_train, y_train) -> Pipeline:
    pipeline = build_pipeline()
    logger.info("Training pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")
    return pipeline


def save_model(pipeline: Pipeline, classes) -> None:
    ensure_dirs(MODEL_DIR)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Model saved -> %s", MODEL_PATH)
    save_json({"classes": list(classes)}, LABEL_MAP_PATH)
    logger.info("Label map saved -> %s", LABEL_MAP_PATH)


def load_model() -> Pipeline:
    pipeline = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
    return pipeline
