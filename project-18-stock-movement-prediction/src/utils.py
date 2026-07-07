"""
Utility functions for MarketSentinel system.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"❌ Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"❌ Error parsing YAML: {e}")
        raise


def ensure_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories based on config.
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    for key, path_str in paths.items():
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"✅ Ensured directory: {path}")
    
    # Also create logs directory
    Path("logs").mkdir(exist_ok=True)


def save_model_metadata(ticker: str, metadata: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Save model metadata to JSON file.
    
    Args:
        ticker: Stock ticker symbol
        metadata: Model metadata dictionary
        config: Configuration dictionary
    """
    models_path = Path(config['paths']['models'])
    metadata_path = models_path / f"{ticker}_metadata.json"
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"✅ Saved metadata to {metadata_path}")


def load_model_metadata(ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Load model metadata from JSON file.
    
    Args:
        ticker: Stock ticker symbol
        config: Configuration dictionary
        
    Returns:
        Metadata dictionary
    """
    models_path = Path(config['paths']['models'])
    metadata_path = models_path / f"{ticker}_metadata.json"
    
    if not metadata_path.exists():
        logger.warning(f"⚠️  Metadata not found for {ticker}")
        return {}
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def format_metric(value: float, metric_name: str = "", decimals: int = 4) -> str:
    """Format metric value for display.
    
    Args:
        value: Metric value
        metric_name: Name of metric for context
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def print_section(title: str) -> None:
    """Print a formatted section header.
    
    Args:
        title: Section title
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_summary_table(data: Dict[str, Any], title: str = "") -> None:
    """Print formatted summary table.
    
    Args:
        data: Dictionary of metrics
        title: Optional title
    """
    if title:
        print(f"\n📊 {title}")
        print("-" * 50)
    
    for key, value in data.items():
        if isinstance(value, float):
            print(f"  {key:.<40} {value:>8.4f}")
        else:
            print(f"  {key:.<40} {str(value):>8}")


def validate_ticker(ticker: str, available_tickers: list) -> bool:
    """Validate ticker symbol.
    
    Args:
        ticker: Ticker to validate
        available_tickers: List of available tickers
        
    Returns:
        True if valid, False otherwise
    """
    if ticker.upper() not in available_tickers:
        logger.warning(f"⚠️  Ticker {ticker} not in available list")
        return False
    return True
