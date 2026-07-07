"""Small shared helpers used across the project."""

from pathlib import Path
import logging
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the project YAML config into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict) -> None:
    """Create every output directory referenced in the config if missing."""
    Path(config["paths"]["models_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["reports_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["figures_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["data"]).parent.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a configured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                               datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
