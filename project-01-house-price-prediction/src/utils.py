from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Save a dictionary to a JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
