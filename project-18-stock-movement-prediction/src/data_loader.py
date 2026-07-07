"""
Data loading module with Kaggle API integration and synthetic data fallback.
"""

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def download_kaggle_dataset(dataset_name: str, output_path: str) -> bool:
    """Download dataset from Kaggle using Kaggle API.
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'camnugent/sandp500')
        output_path: Where to save downloaded files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        logger.info(f"📥 Attempting to download {dataset_name} from Kaggle...")
        
        # Check if kaggle.json exists
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if not kaggle_json.exists():
            logger.warning("⚠️  kaggle.json not found. Searching project root...")
            if not Path('kaggle.json').exists():
                logger.error("❌ kaggle.json not found. See setup instructions.")
                return False
        
        # Initialize API
        api = KaggleApi()
        api.authenticate()
        
        # Create output directory
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        api.dataset_download_files(dataset_name, path=output_path, unzip=True)
        logger.info(f"✅ Successfully downloaded {dataset_name}")
        return True
        
    except ImportError:
        logger.error("❌ kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"❌ Kaggle download failed: {e}")
        return False


def load_stock_data(
    ticker: str,
    data_path: str = "data/raw/",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Load stock data from CSV files.
    
    Args:
        ticker: Stock ticker symbol
        data_path: Path to raw data directory
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Try to find the CSV file
        data_dir = Path(data_path)
        
        # Look for ticker CSV (various naming conventions)
        candidates = [
            data_dir / f"{ticker}.csv",
            data_dir / f"{ticker.lower()}.csv",
            data_dir / "individual_stocks_5yr" / f"{ticker}.csv",
        ]
        
        csv_file = None
        for candidate in candidates:
            if candidate.exists():
                csv_file = candidate
                break
        
        if not csv_file:
            logger.warning(f"⚠️  Data file for {ticker} not found in {data_path}")
            return pd.DataFrame()
        
        # Load CSV
        df = pd.read_csv(csv_file)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Handle different date column names
        date_col = None
        for col in ['date', 'Date', 'timestamp', 'Timestamp']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df.rename(columns={date_col: 'date'}, inplace=True)
        
        # Ensure required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                # Try case-insensitive match
                matching = [c for c in df.columns if c.lower() == col]
                if matching:
                    df.rename(columns={matching[0]: col}, inplace=True)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Sort chronologically
        df = df.sort_index()
        
        # Filter by date range if provided
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Validate and clean data
        df = _validate_ohlcv_data(df)
        
        logger.info(f"✅ Loaded {len(df)} records for {ticker} from {csv_file.name}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading data for {ticker}: {e}")
        return pd.DataFrame()


def _validate_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Cleaned DataFrame
    """
    original_len = len(df)
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Handle missing values
    if df.isnull().any().any():
        before = len(df)
        df = df.dropna()
        logger.warning(f"⚠️  Dropped {before - len(df)} rows with NaN values")
    
    # Check for invalid prices
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        logger.warning("⚠️  Found non-positive prices, removing affected rows")
        df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
    
    # Ensure OHLC relationships
    df = df[(df['high'] >= df['low']) & 
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])]
    
    logger.info(f"✅ Data validation: kept {len(df)}/{original_len} rows")
    return df


def get_available_tickers(data_path: str = "data/raw/") -> List[str]:
    """Get list of available ticker symbols from data directory.
    
    Args:
        data_path: Path to raw data directory
        
    Returns:
        Sorted list of available tickers
    """
    try:
        data_dir = Path(data_path)
        
        # Find all CSV files
        csv_files = list(data_dir.glob("*.csv")) + \
                   list(data_dir.glob("individual_stocks_5yr/*.csv"))
        
        # Extract ticker symbols
        tickers = set()
        for csv_file in csv_files:
            ticker = csv_file.stem.upper()
            if ticker and ticker != "README":
                tickers.add(ticker)
        
        tickers = sorted(list(tickers))
        logger.info(f"✅ Found {len(tickers)} available tickers")
        return tickers
        
    except Exception as e:
        logger.error(f"❌ Error getting available tickers: {e}")
        return []


def generate_synthetic_stock_data(
    ticker: str,
    n_days: int = 800,
    initial_price: float = 100.0,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """Generate realistic synthetic stock data using GBM with regime switching.
    
    Args:
        ticker: Ticker symbol for synthetic data
        n_days: Number of trading days to generate
        initial_price: Starting price
        config: Configuration dictionary with market regime parameters
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    logger.info(f"🔄 Generating synthetic data for {ticker} ({n_days} days)")
    
    # Default regimes if not provided
    if config is None:
        config = {
            'bull': {'drift': 0.15, 'volatility': 0.18, 'duration_days': [60, 120]},
            'bear': {'drift': -0.10, 'volatility': 0.28, 'duration_days': [40, 90]},
            'sideways': {'drift': 0.02, 'volatility': 0.22, 'duration_days': [50, 100]},
            'crash': {'drift': -0.25, 'volatility': 0.45, 'probability': 0.05}
        }
    
    # Market regimes sequence
    regimes = ['bull', 'sideways', 'bull', 'bear', 'sideways', 'bull', 'bear', 'bull']
    
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    current_date = datetime(2020, 1, 1)
    current_price = initial_price
    regime_idx = 0
    days_in_regime = 0
    
    np.random.seed(42)
    
    for day in range(n_days):
        # Market regime rotation
        current_regime = regimes[regime_idx % len(regimes)]
        regime_config = config.get(current_regime, config['bull'])
        
        regime_duration = np.random.randint(
            regime_config['duration_days'][0],
            regime_config['duration_days'][1]
        )
        
        if days_in_regime >= regime_duration:
            regime_idx += 1
            days_in_regime = 0
            current_regime = regimes[regime_idx % len(regimes)]
            regime_config = config.get(current_regime, config['bull'])
        
        # GBM parameters
        drift = regime_config['drift'] / 252
        volatility = regime_config['volatility'] / np.sqrt(252)
        
        # Generate daily price movement
        z = np.random.normal(0, 1)
        daily_return = drift + volatility * z
        
        # Random intraday volatility for OHLH
        intraday_vol = np.random.uniform(0.005, 0.02)
        
        open_price = current_price
        close_price = open_price * np.exp(daily_return)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, intraday_vol)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, intraday_vol)))
        
        # Volume (correlated with volatility)
        base_volume = 1000000 + np.random.normal(0, 100000)
        volume = int(max(100000, base_volume * (1 + abs(daily_return) * 10)))
        
        dates.append(current_date)
        opens.append(open_price)
        closes.append(close_price)
        highs.append(high_price)
        lows.append(low_price)
        volumes.append(volume)
        
        current_price = close_price
        
        # Skip weekends
        current_date += timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        
        days_in_regime += 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    logger.info(f"✅ Generated synthetic data: {len(df)} days, price range ${df['close'].min():.2f}-${df['close'].max():.2f}")
    
    return df


def save_processed_data(
    df: pd.DataFrame,
    ticker: str,
    output_path: str = "data/processed/"
) -> None:
    """Save processed data to CSV.
    
    Args:
        df: DataFrame to save
        ticker: Ticker symbol
        output_path: Output directory
    """
    try:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        filepath = Path(output_path) / f"{ticker}_features.csv"
        df.to_csv(filepath)
        logger.info(f"✅ Saved processed data to {filepath}")
    except Exception as e:
        logger.error(f"❌ Error saving processed data: {e}")
