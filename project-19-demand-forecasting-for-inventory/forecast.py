#!/usr/bin/env python3
"""
ForecastIQ Forecasting Script

CLI entry point for generating future demand forecasts for a specific store.

Usage:
    python forecast.py --store 1 --days 30
    python forecast.py --store 5 --days 60
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from src.data_loader import load_data
from src.features import engineer_features
from src.model import DemandForecaster
from src.forecaster import forecast_future, scenario_simulation
from src.utils import load_config, setup_logger

logger = setup_logger(__name__)


def main() -> None:
    """Run forecasting pipeline."""
    parser = argparse.ArgumentParser(description="ForecastIQ: Generate demand forecasts")
    parser.add_argument("--store", type=int, default=1, help="Store ID to forecast (default: 1)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to forecast (default: 30)")
    parser.add_argument("--scenario", action="store_true", help="Run 3 promotion scenarios")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🔮 ForecastIQ Forecasting Pipeline")
    logger.info("=" * 60)

    # Load configuration and data
    config = load_config("config.yaml")
    df, data_source = load_data(config)

    # Engineer features
    df_feat = engineer_features(df, config)
    logger.info(f"✅ Features engineered: {len(df_feat):,} rows")

    # Load model
    model_path = config["paths"]["model"]
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("Please run: python train.py")
        sys.exit(1)

    model = DemandForecaster.load(model_path)
    logger.info(f"✅ Model loaded (trained: {model.trained_at})")

    # Forecast
    logger.info("=" * 60)
    logger.info(f"🏪 Store: {args.store} | Horizon: {args.days} days")
    logger.info("=" * 60)

    if args.scenario:
        logger.info("📊 Running 3 promotion scenarios...")
        scenarios = scenario_simulation(model, df_feat, args.store, args.days, config)

        for scenario_name, forecast_df in scenarios.items():
            logger.info(f"\n{scenario_name}:")
            logger.info(f"  Total forecasted sales: €{forecast_df['forecasted_sales'].sum():,.0f}")
            logger.info(f"  Average daily sales: €{forecast_df['forecasted_sales'].mean():,.0f}")

            # Save scenario
            scenario_filename = scenario_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".csv"
            scenario_path = f"reports/{scenario_filename}"
            forecast_df.to_csv(scenario_path, index=False)
            logger.info(f"  Saved to: {scenario_path}")

    else:
        # Single forecast (no promo by default)
        promo_schedule = [0] * args.days
        forecast_df = forecast_future(model, df_feat, args.store, args.days, promo_schedule, config)

        if forecast_df.empty:
            logger.error(f"❌ Could not generate forecast for store {args.store}")
            sys.exit(1)

        logger.info(f"✅ Forecast generated: {len(forecast_df)} days")
        logger.info(f"Total forecasted sales: €{forecast_df['forecasted_sales'].sum():,.0f}")
        logger.info(f"Average daily sales: €{forecast_df['forecasted_sales'].mean():,.0f}")
        logger.info(f"Min/Max: €{forecast_df['forecasted_sales'].min():.0f} / €{forecast_df['forecasted_sales'].max():.0f}")

        # Save forecast
        forecast_path = f"reports/forecast_store_{args.store}.csv"
        forecast_df.to_csv(forecast_path, index=False)
        logger.info(f"\n💾 Forecast saved to: {forecast_path}")

        # Display first few rows
        logger.info("\nForecast sample (first 5 days):")
        logger.info(forecast_df.head().to_string())

    logger.info("\n" + "=" * 60)
    logger.info("✅ Forecasting complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
