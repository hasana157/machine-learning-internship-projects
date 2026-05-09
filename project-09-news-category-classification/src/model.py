"""src/model.py — Pipeline builders and feature extraction."""
import logging
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

def build_logreg_pipeline(max_features=150_000, ngram_range=(1,3), C=5.0,
                           solver="saga", max_iter=1000, random_state=42):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range,
            sublinear_tf=True, min_df=2, max_df=0.95,
            strip_accents="unicode", analyzer="word",
        )),
        ("clf", LogisticRegression(
            C=C, solver=solver, max_iter=max_iter,
            class_weight="balanced", random_state=random_state,
        )),
    ])

def build_svm_pipeline(max_features=150_000, ngram_range=(1,3), C=1.0,
                        max_iter=2000, random_state=42):
    inner = LinearSVC(C=C, max_iter=max_iter, class_weight="balanced",
                      random_state=random_state, loss="squared_hinge", dual=True)
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range,
            sublinear_tf=True, min_df=2, max_df=0.95,
            strip_accents="unicode", analyzer="word",
        )),
        ("clf", CalibratedClassifierCV(inner, cv=3, method="sigmoid")),
    ])

def get_top_features(pipeline, class_names: list, top_n: int = 20) -> Dict[str, list]:
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(classifier, "estimators_"):
        coef = np.mean([est.estimator.coef_ for est in classifier.estimators_], axis=0)
    elif hasattr(classifier, "coef_"):
        coef = classifier.coef_
    else:
        logger.warning("Cannot extract features from this classifier.")
        return {}
    result = {}
    for idx, cls in enumerate(class_names):
        if idx >= coef.shape[0]: break
        top_idx = np.argsort(coef[idx])[::-1][:top_n]
        result[cls] = [(feature_names[i], float(coef[idx][i])) for i in top_idx]
    return result
