"""
CLI script to run inference on a provided CSV file.
Usage: python detect.py --input data/sensor_data.csv
"""

import argparse
import pandas as pd
from pathlib import Path

from src.utils import load_config, setup_logger
from src.model import AnomalyDetector
from src.detector import run_detection

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="SentinelFlow Anomaly Detection CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="reports/detection_results.csv", help="Path to save results")
    args = parser.parse_args()

    config = load_config("config.yaml")
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading data from {input_path}...")
    df_raw = pd.read_csv(input_path)
    
    model_path = config["paths"]["model"]
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}. Please run 'make train' first.")
        return
        
    logger.info("Loading trained AnomalyDetector...")
    detector = AnomalyDetector.load(model_path)
    
    df_results = run_detection(df_raw, detector, config)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    total = len(df_results)
    anomalies = df_results["predicted_label"].sum()
    print(f"\n✅ Detection Complete.")
    print(f"Total points analyzed: {total}")
    print(f"Anomalies detected:    {anomalies} ({anomalies/total*100:.2f}%)")
    print(f"Results saved to:      {output_path}\n")

if __name__ == "__main__":
    main()
