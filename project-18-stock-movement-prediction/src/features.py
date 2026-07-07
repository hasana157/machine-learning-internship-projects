"""
Feature engineering module with 30+ technical indicators.
Ensures NO lookahead bias - all features use only data available at time t.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Engineer all features from OHLCV data.
    
    Args:
        df: OHLCV DataFrame with date index
        config: Configuration dictionary
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Price-based features
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)
    df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_to_open'] = (df['close'] - df['open']) / df['open']
    
    # Moving Averages
    for period in [5, 10, 20, 50, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # Price to MA ratios
    df['price_to_ma_5'] = df['close'] / df['sma_5']
    df['price_to_ma_20'] = df['close'] / df['sma_20']
    df['price_to_ma_50'] = df['close'] / df['sma_50']
    
    # Moving Average Crossovers
    df['ma_crossover_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
    df['ma_crossover_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)
    
    # Momentum Indicators
    df['rsi_14'] = calculate_rsi(df['close'], period=14)
    macd, macd_signal, macd_hist = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_hist
    
    df['roc_10'] = df['close'].pct_change(10)
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    
    # Volatility Features
    df['volatility_5'] = df['return_1d'].rolling(window=5).std()
    df['volatility_20'] = df['return_1d'].rolling(window=20).std()
    df['atr_14'] = calculate_atr(df, period=14)
    
    # Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(df['close'], period=20, std_dev=2.0)
    df['bollinger_upper'] = bb_upper
    df['bollinger_lower'] = bb_lower
    df['bollinger_width'] = (bb_upper - bb_lower) / df['sma_20']
    df['price_position_in_bb'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    # Volume Features
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['volume_price_trend'] = (df['return_1d'] * df['volume']).cumsum()
    df['volume_volatility'] = df['volume'].rolling(window=10).std()
    
    # Stochastic Oscillator
    stoch_k, stoch_d = calculate_stochastic(df, period=14)
    df['stochastic_k'] = stoch_k
    df['stochastic_d'] = stoch_d
    
    # Williams %R
    df['williams_r'] = calculate_williams_r(df, period=14)
    
    # CCI
    df['cci_20'] = calculate_cci(df, period=20)
    
    # Lag Features (NO LOOKAHEAD)
    for lag in [1, 2, 3, 5]:
        df[f'lag_return_{lag}'] = df['return_1d'].shift(lag)
    
    df['lag_volume_ratio_1'] = df['volume_ratio'].shift(1)
    df['lag_volume_ratio_2'] = df['volume_ratio'].shift(2)
    df['lag_rsi_1'] = df['rsi_14'].shift(1)
    
    # Temporal Features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    df['days_since_year_start'] = df.index.dayofyear
    
    # TARGET VARIABLE (CRITICAL - NO LOOKAHEAD)
    # 1 = price goes UP next day, 0 = DOWN/flat
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop rows with NaN (created by rolling/lag/target)
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    
    logger.info(f"✅ Engineered {len(df.columns)} features ({dropped} rows dropped due to NaN)")
    
    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.
    
    Args:
        series: Price series
        period: RSI period
        
    Returns:
        RSI values
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD line, signal line, and histogram.
    
    Args:
        series: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
        
    Returns:
        Tuple of (MACD, Signal, Histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range.
    
    Args:
        df: OHLCV DataFrame
        period: ATR period
        
    Returns:
        ATR values
    """
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Bollinger Bands.
    
    Args:
        series: Price series
        period: MA period
        std_dev: Standard deviation multiplier
        
    Returns:
        Tuple of (upper_band, lower_band)
    """
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower


def calculate_stochastic(
    df: pd.DataFrame,
    period: int = 14,
    k_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator.
    
    Args:
        df: OHLCV DataFrame
        period: Stochastic period
        k_period: K smoothing period
        
    Returns:
        Tuple of (%K, %D)
    """
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
    stoch_d = stoch_k.rolling(window=k_period).mean()
    
    return stoch_k, stoch_d


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R.
    
    Args:
        df: OHLCV DataFrame
        period: Williams %R period
        
    Returns:
        Williams %R values
    """
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    
    williams_r = -100 * (high_max - df['close']) / (high_max - low_min + 1e-10)
    return williams_r


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index.
    
    Args:
        df: OHLCV DataFrame
        period: CCI period
        
    Returns:
        CCI values
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean())
    
    cci = (tp - sma) / (0.015 * mad + 1e-10)
    return cci


def get_feature_names() -> List[str]:
    """Get list of all feature names generated by engineer_features.
    
    Returns:
        List of feature names
    """
    features = [
        # Price features
        'return_1d', 'return_5d', 'return_20d', 'log_return_1d', 'high_low_range', 'close_to_open',
        # Moving Averages
        'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
        'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
        # Price to MA
        'price_to_ma_5', 'price_to_ma_20', 'price_to_ma_50',
        # Crossovers
        'ma_crossover_5_20', 'ma_crossover_50_200',
        # Momentum
        'rsi_14', 'macd', 'macd_signal', 'macd_histogram', 'roc_10', 'momentum_5',
        # Volatility
        'volatility_5', 'volatility_20', 'atr_14',
        # Bollinger Bands
        'bollinger_upper', 'bollinger_lower', 'bollinger_width', 'price_position_in_bb',
        # Volume
        'volume_sma_20', 'volume_ratio', 'obv', 'volume_price_trend', 'volume_volatility',
        # Stochastic & Williams
        'stochastic_k', 'stochastic_d', 'williams_r', 'cci_20',
        # Lag features
        'lag_return_1', 'lag_return_2', 'lag_return_3', 'lag_return_5',
        'lag_volume_ratio_1', 'lag_volume_ratio_2', 'lag_rsi_1',
        # Temporal
        'day_of_week', 'month', 'quarter', 'is_month_start', 'is_month_end', 
        'is_quarter_end', 'days_since_year_start'
    ]
    return features
