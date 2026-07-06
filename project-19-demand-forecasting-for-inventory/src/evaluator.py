"""
Evaluation metrics, error analysis, and report generation.

Computes:
    - MAE, RMSE, MAPE, RMSPE (Kaggle metric), R²
    - Per-store breakdown
    - Top/bottom performers
    - Insight correlations
    - 6+ publication-quality figures
    - Formatted evaluation report
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_pred_rf: np.ndarray, y_pred_baseline: np.ndarray
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for RF and baseline models.

    Metrics:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - MAPE (Mean Absolute Percentage Error)
        - RMSPE (Root Mean Squared Percentage Error) — Kaggle competition metric
        - R² (Coefficient of determination)

    Args:
        y_true: Ground truth sales values.
        y_pred_rf: Random Forest predictions.
        y_pred_baseline: Linear Regression baseline predictions.

    Returns:
        Dictionary with keys:
            rf_mae, rf_rmse, rf_mape, rf_rmspe, rf_r2,
            baseline_mae, baseline_rmse, baseline_mape, baseline_rmspe, baseline_r2
    """
    metrics = {}

    for prefix, y_pred in [("rf", y_pred_rf), ("baseline", y_pred_baseline)]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        rmspe = np.sqrt(np.mean(((y_true - y_pred) / (y_true + 1e-6)) ** 2))
        r2 = r2_score(y_true, y_pred)

        metrics[f"{prefix}_mae"] = mae
        metrics[f"{prefix}_rmse"] = rmse
        metrics[f"{prefix}_mape"] = mape
        metrics[f"{prefix}_rmspe"] = rmspe
        metrics[f"{prefix}_r2"] = r2

    return metrics


def compute_per_store_metrics(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Compute metrics per store.

    Args:
        df: Test DataFrame with store_id and metadata columns.
        y_true: Ground truth sales.
        y_pred: Predictions.

    Returns:
        DataFrame with per-store metrics:
            [store_id, mae, rmse, mape, rmspe, r2, n_rows, mean_sales, std_sales,
             store_type, assortment, competition_distance]
    """
    store_metrics = []

    for store_id in df["store_id"].unique():
        mask = df["store_id"] == store_id
        y_true_store = y_true[mask]
        y_pred_store = y_pred[mask]
        df_store = df[mask]

        if len(y_true_store) == 0:
            continue

        mae = mean_absolute_error(y_true_store, y_pred_store)
        rmse = np.sqrt(mean_squared_error(y_true_store, y_pred_store))
        mape = np.mean(np.abs((y_true_store - y_pred_store) / (y_true_store + 1e-6))) * 100
        rmspe = np.sqrt(np.mean(((y_true_store - y_pred_store) / (y_true_store + 1e-6)) ** 2))
        r2 = r2_score(y_true_store, y_pred_store)

        store_type = df_store["store_type"].iloc[0]
        assortment = df_store["assortment"].iloc[0]
        comp_dist = df_store["competition_distance"].iloc[0]

        store_metrics.append({
            "store_id": store_id,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "rmspe": rmspe,
            "r2": r2,
            "n_rows": len(y_true_store),
            "mean_sales": y_true_store.mean(),
            "std_sales": y_true_store.std(),
            "store_type": store_type,
            "assortment": assortment,
            "competition_distance": comp_dist,
        })

    return pd.DataFrame(store_metrics)


def generate_evaluation_figures(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_rf: np.ndarray,
    y_pred_baseline: np.ndarray,
    model,
    metrics: Dict[str, float],
    output_dir: str,
) -> None:
    """
    Generate 6 publication-quality evaluation figures.

    Figures:
        1. forecast_sample_grid.png — 3×3 grid of sample store forecasts
        2. store_error_distribution.png — histogram of per-store MAE
        3. feature_importance.png — top 20 features with category colors
        4. model_comparison.png — RF vs Linear Regression metrics
        5. sales_by_storetype.png — box plot by store type with error overlay
        6. evaluation_report.txt — formatted text report

    Args:
        df_test: Test DataFrame.
        y_test: Ground truth sales.
        y_pred_rf: RF predictions.
        y_pred_baseline: Baseline predictions.
        model: Trained DemandForecaster.
        metrics: Dict of computed metrics.
        output_dir: Output directory for figures.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sns.set_style("darkgrid")
    sns.set_palette("husl")

    # Figure 1: Forecast sample grid (3×3)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("Sample Store Forecasts — Actual vs RF vs Baseline", fontsize=16, y=1.00)

    store_ids = df_test["store_id"].unique()[:9] if len(df_test["store_id"].unique()) >= 9 else df_test["store_id"].unique()

    for idx, ax in enumerate(axes.flat):
        if idx >= len(store_ids):
            ax.remove()
            continue

        store_id = store_ids[idx]
        mask = df_test["store_id"] == store_id

        dates = df_test.loc[mask, "date"].values
        actual = y_test[mask]
        pred_rf = y_pred_rf[mask]
        pred_base = y_pred_baseline[mask]

        mae_val = mean_absolute_error(actual, pred_rf)
        store_type = df_test.loc[mask, "store_type"].iloc[0]

        ax.plot(dates, actual, "b-", label="Actual", linewidth=2)
        ax.plot(dates, pred_rf, "o--", color="orange", label="RF", alpha=0.7)
        ax.plot(dates, pred_base, "^:", color="gray", label="Baseline", alpha=0.6)

        ax.set_title(f"Store {store_id} | MAE: {mae_val:.0f} | Type: {store_type}", fontsize=10)
        ax.set_ylabel("Sales (€)")
        ax.tick_params(axis="x", rotation=45)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/forecast_sample_grid.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("✅ Saved: forecast_sample_grid.png")

    # Figure 2: Store error distribution
    store_metrics = compute_per_store_metrics(df_test, y_test, y_pred_rf)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(store_metrics["mae"], bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(store_metrics["mae"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {store_metrics['mae'].mean():.0f}")
    ax.axvline(store_metrics["mae"].median(), color="green", linestyle="--", linewidth=2, label=f"Median: {store_metrics['mae'].median():.0f}")
    ax.set_xlabel("MAE (€)", fontsize=12)
    ax.set_ylabel("Number of Stores", fontsize=12)
    ax.set_title("Distribution of Forecast Error Across Stores", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/store_error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("✅ Saved: store_error_distribution.png")

    # Figure 3: Feature importance
    feature_imp = model.get_feature_importances().head(20)

    feature_colors = {
        "lag": "#378ADD",
        "roll": "#1D9E75",
        "calendar": "#639922",
        "store": "#D85A30",
        "promo": "#F59E0B",
    }

    colors = []
    for feat in feature_imp.index:
        if "lag" in feat:
            colors.append(feature_colors["lag"])
        elif "roll" in feat or "ewm" in feat:
            colors.append(feature_colors["roll"])
        elif any(cal in feat for cal in ["day", "week", "month", "quarter", "year", "christmas", "easter", "weekend", "is_month"]):
            colors.append(feature_colors["calendar"])
        elif any(store in feat for store in ["store_id", "competition", "type", "assortment"]):
            colors.append(feature_colors["store"])
        elif "promo" in feat:
            colors.append(feature_colors["promo"])
        else:
            colors.append("#999999")

    fig, ax = plt.subplots(figsize=(12, 8))
    feature_imp.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Top 20 Feature Importances", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("✅ Saved: feature_importance.png")

    # Figure 4: Model comparison
    metric_names = ["MAE", "RMSE", "MAPE", "RMSPE"]
    rf_vals = [
        metrics["rf_mae"],
        metrics["rf_rmse"],
        metrics["rf_mape"],
        metrics["rf_rmspe"],
    ]
    baseline_vals = [
        metrics["baseline_mae"],
        metrics["baseline_rmse"],
        metrics["baseline_mape"],
        metrics["baseline_rmspe"],
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, rf_vals, width, label="Random Forest", color="steelblue")
    bars2 = ax.bar(x + width / 2, baseline_vals, width, label="Linear Regression", color="coral")

    ax.set_ylabel("Error Value", fontsize=12)
    ax.set_title("Model Comparison: RF vs Linear Regression", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("✅ Saved: model_comparison.png")

    # Figure 5: Sales by store type
    fig, ax = plt.subplots(figsize=(12, 6))
    store_types = df_test["store_type"].unique()
    store_type_sales = [df_test[df_test["store_type"] == st]["sales"] if "sales" in df_test.columns else y_test[df_test["store_type"] == st] for st in store_types]

    ax.boxplot(store_type_sales, labels=store_types)
    ax.set_ylabel("Daily Sales (€)", fontsize=12)
    ax.set_xlabel("Store Type", fontsize=12)
    ax.set_title("Sales Distribution by Store Type", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sales_by_storetype.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("✅ Saved: sales_by_storetype.png")


def generate_evaluation_report(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_rf: np.ndarray,
    metrics: Dict[str, float],
    data_source: str,
    output_dir: str,
) -> None:
    """
    Generate formatted evaluation report.

    Args:
        df_test: Test DataFrame.
        y_test: Ground truth.
        y_pred_rf: RF predictions.
        metrics: Metrics dictionary.
        data_source: "kaggle" or "synthetic".
        output_dir: Output directory.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report = []
    report.append("┌─────────────────────────────────────────────────┐")
    report.append("│  ForecastIQ — Evaluation Report                 │")
    report.append(f"│  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'):.<39}│")
    report.append(f"│  Data Source: {data_source.ljust(29):.<39}│")
    report.append("├─────────────────────────────────────────────────┤")
    report.append("│  Dataset Stats                                  │")
    report.append(f"│    Stores:     {df_test['store_id'].nunique():<25}         │")
    report.append(f"│    Date range: {df_test['date'].min().strftime('%Y-%m-%d')} → {df_test['date'].max().strftime('%Y-%m-%d'):.<14}│")
    report.append(f"│    Test rows:  {len(df_test):,}                       │")
    report.append("├─────────────────────────────────────────────────┤")
    report.append("│  Model Performance                              │")
    report.append("│                RF          Linear Baseline      │")
    report.append(f"│  MAE         {metrics['rf_mae']:>8.2f}        {metrics['baseline_mae']:>8.2f}             │")
    report.append(f"│  RMSE        {metrics['rf_rmse']:>8.2f}        {metrics['baseline_rmse']:>8.2f}             │")
    report.append(f"│  MAPE        {metrics['rf_mape']:>7.2f}%       {metrics['baseline_mape']:>7.2f}%            │")
    report.append(f"│  RMSPE       {metrics['rf_rmspe']:>8.4f}        {metrics['baseline_rmspe']:>8.4f}   ← Kaggle  │")
    report.append(f"│  R²          {metrics['rf_r2']:>8.4f}        {metrics['baseline_r2']:>8.4f}             │")
    report.append("├─────────────────────────────────────────────────┤")
    report.append("│  Key Insights                                   │")

    store_metrics = compute_per_store_metrics(df_test, y_test, y_pred_rf)
    hardest_stores = store_metrics.nlargest(5, "mae")

    report.append("│  Top 5 Hardest Stores:                          │")
    for _, row in hardest_stores.iterrows():
        report.append(f"│    Store {row['store_id']} ({row['store_type']}): MAE={row['mae']:.0f}, RMSPE={row['rmspe']:.4f}")

    report.append("│                                                 │")
    rf_improvement = (
        (metrics["baseline_mape"] - metrics["rf_mape"]) / metrics["baseline_mape"] * 100
    )
    report.append(f"│  RF beats baseline MAPE by: {rf_improvement:.1f}%")
    report.append("│                                                 │")
    report.append("└─────────────────────────────────────────────────┘")

    report_text = "\n".join(report)
    report_path = f"{output_dir}/evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"✅ Evaluation report saved to {report_path}")
    logger.info("\n" + report_text)
