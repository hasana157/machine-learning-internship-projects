from __future__ import annotations

import argparse
import json
import joblib
import pandas as pd

def main(model_path: str, input_json: str) -> None:
    """
    input_json can now be a path to a JSON file, e.g. input.json
    """
    pipeline = joblib.load(model_path)

    # Correct indentation here!
    with open(input_json, "r") as f:
         row = json.load(f)

    X = pd.DataFrame([row])
    pred = pipeline.predict(X)[0]
    print(f"Predicted price (target units): {pred:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    args = parser.parse_args()

    main(args.model_path, args.input_json)
