#!/usr/bin/env python3
"""
MarketSentinel Backtesting CLI

Simulate trading strategies based on model predictions.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from src.utils import load_config, ensure_directories, print_section
from src.data_loader import load_stock_data
from src.trainer import load_model_artifacts
from src.backtester import BacktestEngine
from src.evaluator import plot_equity_curve, plot_drawdown

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main backtesting function."""
    
    parser = argparse.ArgumentParser(
        description="Backtest MarketSentinel trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py --ticker AAPL
  python backtest.py --ticker GOOGL --start-date 2023-01-01 --end-date 2024-01-01
  python backtest.py --ticker TSLA --confidence-threshold 0.7
        """
    )
    
    parser.add_argument('--ticker', required=True, help='Stock ticker symbol')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='Trading confidence threshold (0.0-1.0)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    print_section("💰 MarketSentinel - Backtesting Pipeline")
    
    # Load configuration
    logger.info(f"📖 Loading configuration from {args.config}")
    config = load_config(args.config)
    ensure_directories(config)
    
    ticker = args.ticker.upper()
    
    logger.info(f"🎯 Backtesting for ticker: {ticker}")
    logger.info(f"   Confidence threshold: {args.confidence_threshold:.2f}")
    
    # Load predictions
    pred_path = Path(config['paths']['data_processed']) / f"{ticker}_predictions.csv"
    
    if not pred_path.exists():
        logger.error(f"❌ Predictions not found: {pred_path}")
        logger.info(f"💡 Run 'python train.py --ticker {ticker}' first")
        sys.exit(1)
    
    predictions_df = pd.read_csv(pred_path)
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    
    logger.info(f"✅ Loaded {len(predictions_df)} predictions")
    
    # Load price data
    logger.info(f"📈 Loading price data for {ticker}")
    prices_df = load_stock_data(ticker, config['paths']['data_raw'])
    
    if prices_df.empty:
        logger.error(f"❌ No price data found for {ticker}")
        sys.exit(1)
    
    # Filter by date range if provided
    if args.start_date or args.end_date:
        if args.start_date:
            mask_start = predictions_df['date'] >= args.start_date
        else:
            mask_start = True
        
        if args.end_date:
            mask_end = predictions_df['date'] <= args.end_date
        else:
            mask_end = True
        
        predictions_df = predictions_df[mask_start & mask_end]
    
    logger.info(f"📊 Backtesting period: {predictions_df['date'].min().date()} to {predictions_df['date'].max().date()}")
    
    # Update confidence threshold in config for backtester
    config['backtesting']['confidence_threshold'] = args.confidence_threshold
    
    # Run backtest
    start_time = datetime.now()
    
    backtester = BacktestEngine(
        config['backtesting']['initial_capital'],
        config
    )
    
    results = backtester.run_backtest(predictions_df, prices_df, ticker)
    
    # Save results
    reports_path = Path(config['paths']['reports'])
    reports_path.mkdir(parents=True, exist_ok=True)
    
    # Trade log
    trade_log = backtester.get_trade_log()
    if not trade_log.empty:
        trade_log.to_csv(reports_path / f"{ticker}_trades.csv", index=False)
        logger.info(f"✅ Saved trade log to {reports_path / f'{ticker}_trades.csv'}")
    
    # Equity curve
    equity_curve = backtester.get_equity_curve()
    equity_curve.to_csv(reports_path / f"{ticker}_equity_curve.csv")
    
    # Buy and hold benchmark
    if not prices_df.empty:
        # Align dates
        buy_hold_values = []
        buy_hold_dates = []
        
        initial_capital = config['backtesting']['initial_capital']
        start_price = prices_df.iloc[0]['close']
        
        for date, price in prices_df['close'].items():
            value = initial_capital * (price / start_price)
            buy_hold_values.append(value)
            buy_hold_dates.append(date)
        
        benchmark_series = pd.Series(buy_hold_values, index=buy_hold_dates)
        
        # Plot
        figures_path = Path(config['paths']['figures'])
        figures_path.mkdir(parents=True, exist_ok=True)
        
        plot_equity_curve(equity_curve, benchmark_series, ticker,
                         str(figures_path / f"{ticker}_equity_curve.png"))
        plot_drawdown(equity_curve, ticker,
                     str(figures_path / f"{ticker}_drawdown.png"))
    
    # Summary report
    summary_report = f"""
╔════════════════════════════════════════════════════════════╗
║              BACKTEST SUMMARY REPORT - {ticker}
╚════════════════════════════════════════════════════════════╝

Period:                {predictions_df['date'].min().date()} to {predictions_df['date'].max().date()}

📊 PERFORMANCE METRICS:
  Initial Capital:     ${results['initial_capital']:,.2f}
  Final Value:         ${results['final_value']:,.2f}
  Strategy Return:     {results['total_return_pct']:.2f}%
  
  Buy & Hold Return:   {results['buy_hold_return_pct']:.2f}%
  Buy & Hold Final:    ${results['buy_hold_final_value']:,.2f}
  
  Outperformance:      {results['total_return_pct'] - results['buy_hold_return_pct']:.2f}%

💹 RISK METRICS:
  Max Drawdown:        {results['max_drawdown_pct']:.2f}%
  Sharpe Ratio:        {results['sharpe_ratio']:.2f}

📈 TRADE STATISTICS:
  Number of Trades:    {results['num_trades']}
  Winning Trades:      {results['winning_trades']}
  Win Rate:            {results['win_rate_pct']:.2f}%
  Avg Trade Return:    {results['avg_trade_return_pct']:.2f}%

═══════════════════════════════════════════════════════════════

⚠️  DISCLAIMER: This is an EDUCATIONAL demonstration.
   NOT financial advice. Do not use for actual trading.
"""
    
    print(summary_report)
    
    # Save report
    with open(reports_path / f"{ticker}_backtest_report.txt", 'w') as f:
        f.write(summary_report)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"⏱️  Backtest completed in {elapsed:.2f} seconds")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
