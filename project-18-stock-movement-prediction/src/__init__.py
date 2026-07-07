"""
MarketSentinel - AI-Powered Stock Movement Prediction System

Educational stock prediction system using machine learning with walk-forward
validation and comprehensive backtesting engine.
"""

__version__ = "1.0.0"
__author__ = "MarketSentinel Team"

import logging
import logging.handlers
from pathlib import Path

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("marketsentinel")
logger.setLevel(logging.INFO)

# File handler
fh = logging.handlers.RotatingFileHandler(
    LOG_DIR / "marketsentinel.log",
    maxBytes=10485760,  # 10MB
    backupCount=5
)
fh.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
