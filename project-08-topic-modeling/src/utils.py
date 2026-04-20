"""
utils.py — Shared utility functions for the topic modeling pipeline.

Provides logging configuration, path resolution, config loading,
and reproducibility helpers used across all modules.
"""

import logging
import os
import random
import json
from pathlib import Path

import numpy as np


# ── Project root (two levels up from src/) ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Standard directory layout ─────────────────────────────────────────────────
DIRS = {
    "raw":      PROJECT_ROOT / "data" / "raw",
    "interim":  PROJECT_ROOT / "data" / "interim",
    "processed":PROJECT_ROOT / "data" / "processed",
    "models":   PROJECT_ROOT / "models",
    "figures":  PROJECT_ROOT / "reports" / "figures",
    "reports":  PROJECT_ROOT / "reports",
}


def ensure_dirs() -> None:
    """Create all standard project directories if they do not yet exist."""
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger with a consistent format.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:                       # avoid duplicate handlers
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: Path | str | None = None) -> dict:
    """
    Load project configuration from a JSON file.

    Falls back to sensible defaults when no file is supplied.

    Parameters
    ----------
    config_path : path-like, optional
        Path to a JSON config file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    defaults = {
        "n_topics": 10,
        "model_type": "nmf",          # "nmf" | "lda"
        "max_features": 5000,
        "vectorizer": "tfidf",        # "tfidf" | "count"
        "max_df": 0.95,
        "min_df": 2,
        "n_top_words": 15,
        "random_state": 42,
        "nmf": {
            "init": "nndsvda",
            "max_iter": 400,
            "alpha_W": 0.1,
            "alpha_H": 0.1,
        },
        "lda": {
            "max_iter": 20,
            "learning_method": "online",
            "learning_offset": 50.0,
        },
        "categories": None,           # None → all 20 newsgroups
        "subset": "all",              # "train" | "test" | "all"
    }

    if config_path is None:
        return defaults

    config_path = Path(config_path)
    if not config_path.exists():
        get_logger(__name__).warning(
            "Config file %s not found — using defaults.", config_path
        )
        return defaults

    with open(config_path) as fh:
        user_cfg = json.load(fh)

    return {**defaults, **user_cfg}


def path_for(key: str, filename: str = "") -> Path:
    """
    Convenience function to build a file path inside a known project directory.

    Parameters
    ----------
    key : str
        One of the keys in ``DIRS``.
    filename : str
        File name to append (optional).

    Returns
    -------
    Path
    """
    base = DIRS[key]
    return base / filename if filename else base
