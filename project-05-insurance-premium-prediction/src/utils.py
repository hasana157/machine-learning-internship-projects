from pathlib import Path
import json

def ensure_dir(path):
    """Ensures a directory exists, creating it if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(path, payload):
    """Saves a dictionary as a formatted JSON file."""
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)