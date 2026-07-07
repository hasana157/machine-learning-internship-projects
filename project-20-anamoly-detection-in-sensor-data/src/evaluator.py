"""
Evaluation module for SentinelFlow.
Generates performance metrics and visualization reports.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_config, setup_logger
from src.model import AnomalyDetector
from src.detector import run_detection

logger = setup_logger(__name__)

def evaluate_model():
    """
    Run full evaluation pipeline: load test data, predict, calculate metrics, and generate plots.
    """
    config = load_config("config.yaml")
    
    # Setup paths
    reports_dir = Path(config["paths"]["reports"])
    figures_dir = Path(config["paths"]["figures"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data and Model
    data_path = Path(config["paths"]["data"])
    model_path = Path(config["paths"]["model"])
    
    if not data_path.exists() or not model_path.exists():
        logger.error("Data or model not found. Run 'make train' first.")
        return
        
    df_raw = pd.read_csv(data_path)
    if "timestamp" in df_raw.columns:
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        
    detector = AnomalyDetector.load(str(model_path))
    
    # Run Detection
    logger.info("Running detection on dataset for evaluation...")
    df_res = run_detection(df_raw, detector, config)
    
    # We need to evaluate on the test split
    train_split = config["evaluation"]["train_split"]
    split_idx = int(len(df_res) * train_split)
    
    df_test = df_res.iloc[split_idx:].copy()
    
    if "is_anomaly" not in df_test.columns:
        logger.warning("No ground truth 'is_anomaly' column found. Skipping evaluation metrics.")
        return
        
    y_true = df_test["is_anomaly"].values
    y_pred = df_test["predicted_label"].values
    y_scores = df_test["anomaly_score"].values
    y_pred_z = df_test["zscore_label"].values
    
    # 1. Calculate Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Z-Score baseline metrics
    z_auc = roc_auc_score(y_true, df_test[[c for c in df_test.columns if c.startswith("z_")]].abs().max(axis=1))
    
    report_text = f"""
┌──────────────────┬────────┐
│ Metric           │ Score  │
├──────────────────┼────────┤
│ Precision        │ {precision:.4f} │
│ Recall           │ {recall:.4f} │
│ F1-Score         │ {f1:.4f} │
│ AUC-ROC          │ {auc:.4f} │
│ False Alarm Rate │ {far:.4f} │
└──────────────────┴────────┘
"""
    print(report_text)
    with open(reports_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
        
    # Set plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # 2. ANOMALY TIMELINE PLOT
    logger.info("Generating Anomaly Timeline Plot...")
    sensors = ["temp", "vibration", "pressure", "current"]
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    
    for ax, sensor in zip(axes, sensors):
        ax.plot(df_test["timestamp"], df_test[sensor], color='tab:blue', alpha=0.7, label='Normal')
        
        # Detected anomalies (True Positives + False Positives)
        detected = df_test[df_test["predicted_label"] == 1]
        ax.scatter(detected["timestamp"], detected[sensor], color='red', s=30, label='Detected Anomaly', zorder=5)
        
        # Missed anomalies (False Negatives)
        missed = df_test[(df_test["is_anomaly"] == 1) & (df_test["predicted_label"] == 0)]
        if len(missed) > 0:
            ax.scatter(missed["timestamp"], missed[sensor], color='orange', s=30, marker='X', label='Missed (FN)', zorder=6)
            
        # Shaded band for +- 2 std dev (using rolling stats from features)
        if f"{sensor}_rolling_mean" in df_test.columns:
            mean = df_test[f"{sensor}_rolling_mean"]
            std = df_test[f"{sensor}_rolling_std"]
            ax.fill_between(df_test["timestamp"], mean - 2*std, mean + 2*std, color='gray', alpha=0.2, label='±2σ Normal Range')
            
        sensor_tp = len(df_test[(df_test["is_anomaly"] == 1) & (df_test["predicted_label"] == 1) & (df_test[sensor] > df_test[f"{sensor}_rolling_mean"] + 2*df_test[f"{sensor}_rolling_std"])]) # Approximation for title
        total_sensor_anom = len(df_test[(df_test["is_anomaly"] == 1) & (df_test[sensor] > df_test[f"{sensor}_rolling_mean"] + 2*df_test[f"{sensor}_rolling_std"])])
        
        ax.set_title(f"{sensor.capitalize()} Sensor (Detection Stats Approx: {sensor_tp} / {total_sensor_anom})")
        ax.legend(loc='upper right', fontsize='small')
        
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(figures_dir / "anomaly_timeline.png", dpi=300)
    plt.close()
    
    # 3. ANOMALY SCORE DISTRIBUTION
    logger.info("Generating Score Distribution Plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_test, x="anomaly_score", hue="is_anomaly", bins=50, kde=True, 
                 palette={0: "tab:blue", 1: "tab:red"}, alpha=0.5)
    
    # We estimate the normalized threshold from the training threshold
    # Since we don't store the exact normalized threshold, we draw a line at a reasonable cutoff
    # or calculate the F1-optimal threshold on test set for visualization
    precisions, recalls, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(2 * (precisions * recalls) / (precisions + recalls + 1e-8))
    viz_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
    
    plt.axvline(x=viz_threshold, color='k', linestyle='--', label=f'Estimated Optimal Threshold = {viz_threshold:.2f}')
    plt.title("Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "score_distribution.png", dpi=300)
    plt.close()
    
    # 4. ROC CURVE
    logger.info("Generating ROC Curve...")
    fpr_if, tpr_if, _ = roc_curve(y_true, y_scores)
    
    z_scores_max = df_test[[c for c in df_test.columns if c.startswith("z_")]].abs().max(axis=1)
    fpr_z, tpr_z, _ = roc_curve(y_true, z_scores_max)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_if, tpr_if, label=f'Isolation Forest (AUC = {auc:.3f})', color='tab:blue', linewidth=2)
    plt.plot(fpr_z, tpr_z, label=f'Z-Score Baseline (AUC = {z_auc:.3f})', color='tab:orange', linestyle='--', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png", dpi=300)
    plt.close()
    
    # 5. FEATURE IMPORTANCE BAR CHART
    logger.info("Generating Feature Importance Plot...")
    importances = detector.get_feature_importances()
    top_n = 10
    top_features = importances.head(top_n)
    
    colors = []
    for feat in top_features.index:
        if "temp" in feat: colors.append("tab:red")
        elif "vibration" in feat: colors.append("tab:orange")
        elif "pressure" in feat: colors.append("tab:blue")
        elif "current" in feat: colors.append("tab:green")
        else: colors.append("gray")
        
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_features.values, y=top_features.index, palette=colors)
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Permutation Importance")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=300)
    plt.close()
    
    # 6. CSV REPORT
    csv_cols = ["timestamp", "temp", "vibration", "pressure", "current", 
                "anomaly_score", "predicted_label", "is_anomaly"]
    
    # Add dummy anomaly_type for the report if we can't extract it cleanly from generation script 
    # (Since labels were only returned in data_generator and not stored in CSV)
    # We will just write the columns we have
    df_out = df_test[csv_cols].copy()
    df_out.rename(columns={"is_anomaly": "true_label"}, inplace=True)
    df_out.to_csv(reports_dir / "detection_results.csv", index=False)
    
    logger.info("Evaluation complete. All reports and figures generated.")

if __name__ == "__main__":
    evaluate_model()
