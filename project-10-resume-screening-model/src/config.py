"""
config.py
---------
Central configuration for the Resume Screening pipeline.
All paths, hyperparameters, and constants live here.
"""

import os

# Paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

DATASET_PATH   = os.path.join(DATA_DIR,   "resume_dataset.csv")
MODEL_PATH     = os.path.join(MODEL_DIR,  "resume_classifier.joblib")
METRICS_PATH   = os.path.join(REPORT_DIR, "metrics.json")
LABEL_MAP_PATH = os.path.join(MODEL_DIR,  "label_classes.json")

# Feature columns
TEXT_COLUMN  = "resume_text"
LABEL_COLUMN = "JobRole"

# Allowed job-role labels
ALLOWED_LABELS = [
    "Data Analyst",
    "Data Scientist",
    "ML Engineer",
    "Project Manager",
    "Software Engineer",
]

# Train / validation split
TEST_SIZE    = 0.20
RANDOM_STATE = 42
STRATIFY     = True

# TF-IDF hyperparameters
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MIN_DF       = 1

# Logistic Regression hyperparameters
LR_C        = 1.0
LR_MAX_ITER = 2000
LR_SOLVER   = "lbfgs"
