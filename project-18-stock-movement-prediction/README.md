# 📈 MarketSentinel - AI-Powered Stock Movement Prediction

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.1-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-red.svg)
![Kaggle](https://img.shields.io/badge/data-Kaggle-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

---

## ⚠️ IMPORTANT DISCLAIMER

**This project is for EDUCATIONAL PURPOSES ONLY.**

**This is NOT financial advice. Do not use this system for actual trading or investment decisions.**

Stock markets are inherently unpredictable, and past performance does not guarantee future results. The creators of this project assume **NO LIABILITY** for any financial losses incurred from using this tool.

**Always consult with a qualified financial advisor before making investment decisions.**

---

## 🎯 Project Description

**MarketSentinel** is a production-grade machine learning system that predicts next-day stock price direction (UP/DOWN) using:

- **30+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic, and more
- **Walk-Forward Validation**: Prevents data leakage with realistic performance estimation
- **Random Forest Classification**: Ensemble learning with optimized hyperparameters
- **Comprehensive Backtesting**: Trading simulation with transaction costs
- **Interactive Dashboard**: Real-time Streamlit app with Plotly visualizations

### Key Features

✨ **Real Stock Data**: Uses Kaggle S&P 500 dataset (or synthetic fallback)  
🌲 **Random Forest + Logistic Regression**: Primary model + baseline comparison  
📊 **30+ Features**: Price, volume, momentum, volatility, temporal  
🔄 **Walk-Forward Validation**: No lookahead bias - realistic performance  
💰 **Backtesting Engine**: Simulate trading with transaction costs  
📈 **Interactive Dashboard**: 6-page Streamlit app  
🎯 **Confidence Scoring**: Only trade when model is confident  
📁 **Multi-Ticker Support**: Train on any stock  
📊 **Comprehensive Metrics**: Accuracy, Sharpe, max drawdown, ROC-AUC  
🔧 **Fully Configurable**: All parameters in `config.yaml`  

---

## 📁 Project Structure

```
MarketSentinel/
├── app/
│   └── streamlit_app.py              # 6-page interactive dashboard
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Kaggle API + synthetic data
│   ├── features.py                   # 30+ technical indicators
│   ├── model.py                      # Random Forest + Logistic Regression
│   ├── trainer.py                    # Walk-forward validation
│   ├── backtester.py                 # Trading simulation
│   ├── evaluator.py                  # Metrics & visualization
│   └── utils.py                      # Utilities and helpers
├── data/
│   ├── raw/                          # Kaggle CSV files
│   └── processed/                    # Feature-engineered data
├── models/                           # Trained models (.joblib)
├── reports/                          # Evaluation reports & figures
├── config.yaml                       # All configuration
├── train.py                          # Training CLI
├── backtest.py                       # Backtesting CLI
├── requirements.txt                  # Python dependencies
├── Makefile                          # Common commands
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Kaggle API credentials (optional - can use synthetic data)

### 2. Setup

```bash
# Clone repository
git clone https://github.com/yourusername/MarketSentinel.git
cd MarketSentinel

# Install dependencies
pip install -r requirements.txt
# OR use make:
make setup
```

### 3. Setup Kaggle API (OPTIONAL - can use synthetic data)

```bash
# Get API credentials:
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. Download kaggle.json to project root

# Secure credentials (Unix/Mac)
chmod 600 kaggle.json

# Or place in ~/.kaggle/ directory
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Download Data (OPTIONAL)

```bash
# Download S&P 500 data from Kaggle
make download

# OR manually download from:
# https://www.kaggle.com/camnugent/sandp500
# Extract to data/raw/
```

### 5. Train Model

```bash
# Walk-forward validation (recommended)
python train.py --ticker AAPL --mode walk_forward

# OR use make:
make train TICKER=AAPL
```

### 6. Run Backtest

```bash
python backtest.py --ticker AAPL
# OR
make backtest TICKER=AAPL
```

### 7. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
# OR
make app
```

---

## 📚 Detailed Usage

### Training Modes

```bash
# 1. Walk-Forward Validation (RECOMMENDED)
# - Trains periodically, avoids lookahead bias
# - Most realistic performance estimation
python train.py --ticker AAPL --mode walk_forward

# 2. Train-Test Split
# - Simple 80-20 chronological split
# - Quick baseline evaluation
python train.py --ticker AAPL --mode train_test_split

# 3. Full Retrain
# - Train on all available data
# - Best for production deployment
python train.py --ticker AAPL --mode full_retrain
```

### Training with Date Range

```bash
python train.py --ticker AAPL \
  --mode walk_forward \
  --start-date 2021-01-01 \
  --end-date 2023-12-31
```

### Backtesting Options

```bash
# Basic backtest
python backtest.py --ticker AAPL

# Custom confidence threshold
python backtest.py --ticker AAPL --confidence-threshold 0.7

# Specific date range
python backtest.py --ticker AAPL \
  --start-date 2023-01-01 \
  --end-date 2024-01-01
```

### Batch Processing

```bash
# Train multiple tickers
for ticker in AAPL GOOGL MSFT AMZN TSLA; do
    echo "Training $ticker..."
    python train.py --ticker $ticker --mode walk_forward
    python backtest.py --ticker $ticker
done
```

---

## 📥 Dataset Guide

### Option 1: Kaggle S&P 500 Data (Real Data)

**Dataset:** `camnugent/sandp500`  
**URL:** https://www.kaggle.com/camnugent/sandp500

```bash
# Automatic download (requires kaggle.json):
make download

# Manual download:
# 1. Go to Kaggle page
# 2. Download dataset
# 3. Extract CSV files to data/raw/
```

**File Structure:**
```
data/raw/
├── individual_stocks_5yr/
│   ├── AAPL.csv
│   ├── GOOGL.csv
│   └── ... (one CSV per ticker)
└── all_stocks_5yr.csv (combined)
```

### Option 2: Synthetic Data (Fallback)

If Kaggle download fails, the system automatically generates realistic synthetic data with:
- ✓ Multiple market regimes (bull, bear, sideways, crash)
- ✓ Regime switching dynamics
- ✓ Realistic volatility clustering
- ✓ Geometric Brownian Motion pricing

```python
# Synthetic data is auto-generated if real data unavailable
# Configuration in config.yaml under data.synthetic_config
```

### CSV Format Requirements

```
Required columns: date, open, high, low, close, volume

Example:
date,open,high,low,close,volume
2020-01-02,75.09,75.18,74.39,74.95,135647600
2020-01-03,75.00,75.23,74.63,74.81,107525200
```

---

## 🔧 Configuration (config.yaml)

All parameters centralized in `config.yaml` - **zero hardcoded values** in source code.

### Key Settings

```yaml
# Data source
kaggle:
  dataset_name: "camnugent/sandp500"
  fallback_to_synthetic: true

# Model hyperparameters
model:
  random_forest:
    n_estimators: 300      # Number of trees
    max_depth: 15          # Tree depth limit
    min_samples_split: 10  # Min samples to split
    min_samples_leaf: 5    # Min samples per leaf

# Training strategy
training:
  mode: "walk_forward"
  walk_forward:
    initial_window: 252    # First year of training
    retrain_frequency: 20  # Retrain every 20 days

# Backtesting
backtesting:
  initial_capital: 100000
  position_size: 10000
  confidence_threshold: 0.6  # Only trade if confident
  transaction_cost: 0.001    # 0.1% per trade
  strategy_type: "long_only"
```

See `config.yaml` for all available options.

---

## 🏋️ Walk-Forward Validation Explained

**Problem:** Training-test split with time-series allows lookahead bias (using future data to predict)

**Solution:** Walk-forward validation (expanding window, no shuffling)

```
Traditional (WRONG):
[==============TRAIN==============][=====TEST=====]
                                   ↑
                        Uses future data!

Walk-Forward (CORRECT):
[===TRAIN===]→[T]
    [===TRAIN===]→[T]
        [===TRAIN===]→[T]
            [===TRAIN===]→[T]
                      ... continues ...

✓ Train only on past data
✓ Predict one day ahead
✓ Retrain periodically
✓ Realistic performance
```

---

## 📊 Performance Expectations

### Typical Results

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Accuracy | 50-55% | Better than random 50% |
| Precision (UP) | 52-58% | Fewer false positives |
| Sharpe Ratio | 0.2-0.8 | Positive risk-adjusted returns |
| Max Drawdown | 10-20% | Manageable downside |
| Win Rate | 45-55% | % profitable trades |

### Why is Accuracy "Only" 54%?

📌 **Important Context:**

Predicting stock prices is **EXTREMELY difficult** because:

1. **Efficient Market Hypothesis**: Much information is priced in
2. **Random Walk Theory**: Short-term moves have randomness
3. **Non-Stationarity**: Markets change over time
4. **Transaction Costs**: Even 55% accuracy can lose money with high costs

**However**, even 51-52% accuracy is **profitable** if:
- ✓ Confidence filtering is used (only trade high-confidence predictions)
- ✓ Position sizing is appropriate (no over-leveraging)
- ✓ Transaction costs are minimized
- ✓ Stop-losses prevent catastrophic losses

This project demonstrates **ML methodology**, not a get-rich-quick scheme.

---

## 🎓 Feature Engineering

### 30+ Technical Indicators

**Price Features** (6)
- Returns: 1-day, 5-day, 20-day
- Log returns, high-low range, close-to-open

**Moving Averages** (11)
- SMA: 5, 10, 20, 50, 200 period
- EMA: 5, 10, 20, 50, 200 period
- Price-to-MA ratios: 5, 20, 50 period

**Momentum** (6)
- RSI-14, MACD, MACD Signal, MACD Histogram
- Rate of Change (10-period), Momentum (5-period)

**Volatility** (5)
- 5-day and 20-day volatility
- ATR-14, Bollinger Bands (width, position)

**Volume** (5)
- Volume SMA-20, Volume Ratio
- On-Balance Volume (OBV)
- Volume Price Trend, Volume Volatility

**Stochastic & Advanced** (4)
- Stochastic %K, Stochastic %D
- Williams %R, CCI-20

**Lag Features** (7)
- Previous returns (1, 2, 3, 5 days)
- Previous volume ratios & RSI

**Temporal** (7)
- Day of week, month, quarter
- Month/quarter start/end flags
- Days since year start

**Total: 54 features** (all engineered without lookahead bias)

---

## 📈 Dashboard Pages

### 🏠 Dashboard
- Real-time price chart with predictions
- Confusion matrix and accuracy metrics
- Recent predictions table with confidence scores

### 🎯 Predict
- Next-day signal and confidence
- Top 10 feature values
- Feature importance rankings

### 💰 Backtest
- Backtest summary with key metrics
- Equity curve visualization
- Comparison with buy-and-hold

### 📈 Analytics
- ROC curves and metrics comparison
- Feature analysis and distributions
- Market regime performance

### ⚙️ Train
- Model training configuration
- Live progress updates
- Model metadata and status

### ℹ️ About
- Project documentation
- Disclaimer and legal info
- Technical architecture
- Troubleshooting guide

---

## 🧪 Testing & Validation

### Unit Tests (Future)
```bash
pytest tests/ -v
```

### Data Validation
```bash
python -m src.data_loader --validate
```

### Model Validation
```bash
python train.py --ticker SYNTHETIC_TEST --mode train_test_split
```

---

## 🔍 Troubleshooting

### Q: "Kaggle API returns 403 Forbidden"
**A:** Ensure kaggle.json is in correct location and has proper permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Q: "Model accuracy is ~50% (random)"
**A:** This is expected! Stock prediction is hard. Try:
- Increase training window size
- Use confidence filtering (only trade high-confidence predictions)
- Check for data quality issues
- Add more features

### Q: "Walk-forward validation is slow"
**A:** Reduce `retrain_frequency` in config.yaml (e.g., every 30 days instead of 20)

### Q: "Streamlit app crashes on large datasets"
**A:** Enable caching with `@st.cache_data` and reduce date range

### Q: "No data for ticker X"
**A:** 
- Verify ticker is in Kaggle dataset (check `data/raw/` directory)
- Run `make download` to fetch latest data
- Use synthetic data fallback

---

## 📚 Documentation

- [Walk-Forward Validation Guide](docs/walk_forward_guide.md) (future)
- [Feature Engineering Details](docs/features.md) (future)
- [Model Architecture](docs/model_architecture.md) (future)
- [Backtesting Methodology](docs/backtesting.md) (future)

---

## 📦 Dependencies

See `requirements.txt`:
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning
- **streamlit**: Dashboard framework
- **plotly**: Interactive visualizations
- **matplotlib/seaborn**: Static plots
- **joblib**: Model serialization
- **kaggle**: Data source API
- **pyyaml**: Configuration parsing

---

## 📝 License

MIT License - See LICENSE file for details

Copyright (c) 2024 MarketSentinel Contributors

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a pull request

**Code standards:**
- ✓ PEP 8 style guidelines
- ✓ Type hints on all functions
- ✓ Google-style docstrings
- ✓ No hardcoded values (use config.yaml)
- ✓ Tests pass

---

## 🙏 Acknowledgments

- **Kaggle** for S&P 500 dataset
- **scikit-learn** community
- **Streamlit** for amazing framework
- **Plotly** for visualizations

---

## 📞 Support

- 📧 Issues: Open a GitHub issue
- 💬 Discussions: GitHub Discussions tab
- 📚 Documentation: See /docs directory

---

## ⭐ If You Found This Helpful

Please star this repository! It helps others discover the project.

---

**Last Updated:** 2024-01-15  
**Version:** 1.0.0  
**Status:** Active Development

---

### Quick Command Reference

```bash
# Setup
make setup
make download

# Training
make train TICKER=AAPL
make train TICKER=GOOGL MODE=train_test_split

# Backtesting
make backtest TICKER=AAPL

# Dashboard
make app

# Utilities
make clean
make help
```

---

**Remember:** This is an educational project. Always practice proper risk management and never invest money you can't afford to lose!
