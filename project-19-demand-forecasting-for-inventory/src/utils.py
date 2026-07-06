"""
Utility functions: config loading, logging setup, path helpers.

This module provides core utilities for ForecastIQ including YAML configuration
loading, logger initialization, and filesystem path management.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to config.yaml file. Defaults to "config.yaml".

    Returns:
        Dictionary containing all configuration parameters.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If YAML syntax is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def setup_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name (typically __name__).
        log_file: Optional file path to write logs. If None, logs to console only.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure.

    Returns:
        Path object of the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_paths(config: Dict[str, Any]) -> tuple[Path, Path]:
    """
    Get paths to Rossmann Kaggle data files.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (train_csv_path, store_csv_path).
    """
    train_path = Path(config["data"]["kaggle_train"])
    store_path = Path(config["data"]["kaggle_store"])
    return train_path, store_path


def check_kaggle_data_exists(config: Dict[str, Any]) -> bool:
    """
    Check if Kaggle data files exist.

    Args:
        config: Configuration dictionary.

    Returns:
        True if both train.csv and store.csv exist, False otherwise.
    """
    train_path, store_path = get_data_paths(config)
    return train_path.exists() and store_path.exists()
