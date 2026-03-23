from pathlib import Path
import json


def ensure_dir(path):
    """
    Create directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    """
    Save dictionary as JSON file.
    """
    ensure_dir(Path(path).parent)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)