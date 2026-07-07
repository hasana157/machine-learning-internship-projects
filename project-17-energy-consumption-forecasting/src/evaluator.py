"""
Evaluation & reporting: weekday error breakdown, forecast-vs-actual
plot, model comparison bar chart, and a plain-text summary report.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]


def weekday_error_chart(best_result: dict, figures_dir: str):
    error_df = pd.DataFrame({
        "date": pd.to_datetime(best_result["dates_test"]),
        "error": best_result["y_test"] - best_result["preds"],
    })
    error_df["weekday"] = error_df["date"].dt.day_name()

    weekday_mae = (
        error_df.groupby("weekday")["error"]
        .apply(lambda x: x.abs().mean())
        .reindex(WEEKDAY_ORDER)
    )

    plt.figure(figsize=(8, 4))
    weekday_mae.plot(kind="bar", color="#2E86AB")
    plt.ylabel("MAE (kWh)")
    plt.title("Forecast Error by Weekday")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/error_by_weekday.png", dpi=150)
    plt.close()

    return weekday_mae


def forecast_vs_actual_chart(best_result: dict, figures_dir: str):
    dates = pd.to_datetime(best_result["dates_test"])

    plt.figure(figsize=(10, 4.5))
    plt.plot(dates, best_result["y_test"], label="Actual", color="#333333", linewidth=1.5)
    plt.plot(dates, best_result["preds"], label="Forecast", color="#E63946", linewidth=1.5, linestyle="--")
    plt.ylabel("Consumption (kWh)")
    plt.title("Forecast vs Actual — Test Period")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/forecast_vs_actual.png", dpi=150)
    plt.close()


def model_comparison_chart(results: dict, figures_dir: str):
    names = list(results.keys())
    maes = [results[n]["mae"] for n in names]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, maes, color="#457B9D")
    plt.ylabel("MAE (kWh)")
    plt.title("Model Comparison — Test MAE (lower is better)")
    for bar, mae in zip(bars, maes):
        plt.text(bar.get_x() + bar.get_width() / 2, mae, f"{mae:.1f}",
                  ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/model_comparison.png", dpi=150)
    plt.close()


def write_report(best_name: str, metadata: dict, weekday_mae, report_path: str):
    lines = []
    lines.append("=" * 60)
    lines.append("ENERGY CONSUMPTION FORECASTING — EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Best model: {best_name}")
    lines.append("")
    lines.append("Model comparison (test set):")
    for name, m in metadata["metrics"].items():
        marker = " <- selected" if name == best_name else ""
        lines.append(f"  {name:20s} MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  MAPE={m['mape']:.2f}%{marker}")
    lines.append("")
    lines.append("Forecast error by weekday (best model):")
    for day, val in weekday_mae.items():
        lines.append(f"  {day:10s} MAE={val:.2f}")
    lines.append("")
    worst_day = weekday_mae.idxmax()
    lines.append(f"Key insight: highest error occurs on {worst_day}, suggesting the")
    lines.append("model under-captures day-specific behavioural shifts (e.g. weekend")
    lines.append("occupancy or routine changes). Consider day-specific calendar")
    lines.append("features or a separate weekday/weekend model if this gap matters.")
    lines.append("")

    Path(report_path).write_text("\n".join(lines))
