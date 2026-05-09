"""
predict.py
----------
Command-line interface for predicting job category from resume text.

Usage
-----
    # Interactive mode
    python predict.py

    # Pipe mode (JSON output)
    echo "Python, Machine Learning, SQL, 5 years, Masters" | python predict.py

WARNING: FOR EDUCATIONAL PURPOSES ONLY. NOT a hiring tool.
"""

import sys
import json
import logging

from src.predictor import ResumePredictor

logging.basicConfig(level=logging.WARNING)

DISCLAIMER = (
    "\n[!] EDUCATIONAL USE ONLY -- This prediction must NOT be used "
    "for real hiring, recruitment, or candidate evaluation.\n"
)


def run_interactive(predictor: ResumePredictor) -> None:
    print("=" * 60)
    print("  Resume Job-Role Predictor  (Educational Demo)")
    print("=" * 60)
    print(DISCLAIMER)
    print("Enter resume text below, then press Enter twice to predict.")
    print("Type 'quit' to exit.\n")

    while True:
        lines = []
        try:
            while True:
                line = input()
                if line.lower() == "quit":
                    print("Exiting. Goodbye!")
                    return
                if line == "" and lines:
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        text = " ".join(lines).strip()
        if not text:
            continue

        result = predictor.predict(text)
        _print_result(result)


def _print_result(result: dict) -> None:
    print("\n" + "-" * 50)
    print(f"  Predicted Role : {result['predicted_label']}")
    print(f"  Confidence     : {result['confidence'] * 100:.1f}%")
    print()
    print("  All Class Scores:")
    for cls, score in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
        bar = "#" * int(score * 30)
        print(f"    {cls:<20} {score * 100:5.1f}%  {bar}")
    print("-" * 50)
    print(DISCLAIMER)


def run_piped(predictor: ResumePredictor) -> None:
    text = sys.stdin.read().strip()
    if not text:
        print("ERROR: No input text provided.", file=sys.stderr)
        sys.exit(1)
    result = predictor.predict(text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    predictor = ResumePredictor()
    if sys.stdin.isatty():
        run_interactive(predictor)
    else:
        run_piped(predictor)
