"""
utils.py
--------
Shared utility functions: directory setup, JSON I/O, pretty printing.
"""

import json
import os
import logging
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.debug("Directory ensured: %s", d)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Serialize *data* to a JSON file at *path*."""
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("JSON saved → %s", path)


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def section(title: str, width: int = 60) -> None:
    """Print a section separator to stdout."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
