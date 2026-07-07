"""
Backtesting engine for simulating trading strategies.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Trading strategy backtester."""
    
    def __init__(self, initial_capital: float, config: Dict):
        """Initialize backtester.
        
        Args:
            initial_capital: Starting capital in dollars
            config: Configuration dictionary
        """
        self.initial_capital = initial_capital
        self.config = config
        self.current_capital = initial_capital
        self.positions = []  # List of open positions
        self.closed_trades = []
        self.equity_curve = []
        self.equity_dates = []
    
    def run_backtest(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        ticker: str = "UNKNOWN"
    ) -> Dict:
        """Run backtest on predictions.
        
        Args:
            predictions: DataFrame with columns: date, prediction, confidence
            prices: DataFrame with OHLCV data
            ticker: Ticker symbol
            
        Returns:
            Results dictionary
        """
        
        logger.info(f"\n{'='*60}")
        logger.info(f"💰 Running Backtest for {ticker}")
        logger.info(f"{'='*60}")
        
        backtest_config = self.config['backtesting']
        initial_capital = backtest_config['initial_capital']
        position_size = backtest_config['position_size']
        confidence_threshold = backtest_config['confidence_threshold']
        transaction_cost = backtest_config['transaction_cost']
        
        self.current_capital = initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.equity_dates = []
        
        # Merge predictions with prices
        results_df = predictions.copy()
        
        if 'close_price' not in results_df.columns:
            # Try to merge with prices
            prices_daily = prices[['close', 'open']].copy()
            results_df = results_df.join(prices_daily, on='date')
            if 'close' in results_df.columns and 'close_price' not in results_df.columns:
                results_df['close_price'] = results_df['close']
        
        # Portfolio tracking
        current_position = None  # {'type': 'LONG', 'shares': X, 'entry_price': Y, 'entry_date': Z}
        
        for idx, row in results_df.iterrows():
            date = row['date']
            prediction = row.get('prediction_rf', row.get('prediction', 0))
            confidence = row.get('confidence_rf', 0.5)
            price = row.get('close_price', 0)
            
            # Get next day's open price if available
            if isinstance(date, pd.Timestamp):
                date_str = date.date()
            else:
                date_str = date
            
            # Execute trade logic
            signal = prediction  # 1 = UP, 0 = DOWN
            
            if confidence >= confidence_threshold:
                # Only trade if confident
                
                if signal == 1 and current_position is None:
                    # BUY signal
                    shares = int(position_size / price)
                    if shares > 0:
                        cost = shares * price * (1 + transaction_cost)
                        if self.current_capital >= cost:
                            current_position = {
                                'type': 'LONG',
                                'shares': shares,
                                'entry_price': price,
                                'entry_date': date,
                                'entry_cost': cost
                            }
                            self.current_capital -= cost
                            logger.debug(f"  📈 BUY {shares} shares @ ${price:.2f}")
                
                elif signal == 0 and current_position is not None:
                    # SELL signal (close LONG position)
                    if current_position['type'] == 'LONG':
                        exit_value = current_position['shares'] * price * (1 - transaction_cost)
                        pnl = exit_value - current_position['entry_cost']
                        pnl_pct = (pnl / current_position['entry_cost']) * 100
                        
                        self.current_capital += exit_value
                        
                        # Record trade
                        self.closed_trades.append({
                            'entry_date': current_position['entry_date'],
                            'exit_date': date,
                            'type': 'LONG',
                            'entry_price': current_position['entry_price'],
                            'exit_price': price,
                            'shares': current_position['shares'],
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'duration_days': (date - current_position['entry_date']).days
                        })
                        
                        logger.debug(f"  📉 SELL {current_position['shares']} shares @ ${price:.2f}, PnL: ${pnl:.2f}")
                        current_position = None
            
            # Update equity (mark-to-market)
            if current_position is not None:
                position_value = current_position['shares'] * price
                unrealized_pnl = position_value - (current_position['shares'] * current_position['entry_price'])
                total_equity = self.current_capital + unrealized_pnl
            else:
                total_equity = self.current_capital
            
            self.equity_curve.append(total_equity)
            self.equity_dates.append(date)
        
        # Close any remaining position at end
        if current_position is not None:
            last_price = results_df.iloc[-1].get('close_price', current_position['entry_price'])
            exit_value = current_position['shares'] * last_price * (1 - transaction_cost)
            pnl = exit_value - current_position['entry_cost']
            pnl_pct = (pnl / current_position['entry_cost']) * 100
            
            self.current_capital += exit_value
            self.closed_trades.append({
                'entry_date': current_position['entry_date'],
                'exit_date': results_df.iloc[-1]['date'],
                'type': 'LONG',
                'entry_price': current_position['entry_price'],
                'exit_price': last_price,
                'shares': current_position['shares'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'duration_days': (results_df.iloc[-1]['date'] - current_position['entry_date']).days
            })
        
        # Calculate final metrics
        final_value = self.current_capital
        total_return = (final_value - initial_capital) / initial_capital
        
        # Equity curve metrics
        equity_array = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Sharpe ratio (annualized)
        daily_returns = np.diff(equity_array) / equity_array[:-1]
        if len(daily_returns) > 1:
            sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Win rate
        trades_df = pd.DataFrame(self.closed_trades)
        if len(trades_df) > 0:
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = winning_trades / len(trades_df)
            avg_trade_return = trades_df['pnl_pct'].mean()
        else:
            win_rate = 0
            avg_trade_return = 0
        
        # Buy and hold benchmark
        buy_hold_return = (results_df.iloc[-1].get('close_price', 0) / results_df.iloc[0].get('close_price', 1) - 1)
        buy_hold_final = initial_capital * (1 + buy_hold_return)
        
        results = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'buy_hold_return': buy_hold_return,
            'buy_hold_return_pct': buy_hold_return * 100,
            'buy_hold_final_value': buy_hold_final,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(self.closed_trades),
            'winning_trades': winning_trades if len(trades_df) > 0 else 0,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_trade_return_pct': avg_trade_return,
            'ticker': ticker
        }
        
        logger.info(f"\n📊 Backtest Results:")
        logger.info(f"  Strategy Return:    {results['total_return_pct']:.2f}%")
        logger.info(f"  Buy & Hold Return:  {results['buy_hold_return_pct']:.2f}%")
        logger.info(f"  Max Drawdown:       {results['max_drawdown_pct']:.2f}%")
        logger.info(f"  Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
        logger.info(f"  Number of Trades:   {results['num_trades']}")
        logger.info(f"  Win Rate:           {results['win_rate_pct']:.2f}%")
        logger.info(f"  Avg Trade Return:   {results['avg_trade_return_pct']:.2f}%")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as Series.
        
        Returns:
            Series with equity values indexed by date
        """
        return pd.Series(self.equity_curve, index=self.equity_dates)
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get closed trades as DataFrame.
        
        Returns:
            DataFrame with trade details
        """
        return pd.DataFrame(self.closed_trades)
