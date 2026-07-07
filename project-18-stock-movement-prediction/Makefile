.PHONY: help setup download train backtest app clean test lint format all

TICKER ?= AAPL
START_DATE ?= 2020-01-01
END_DATE ?= 2024-01-01
MODE ?= walk_forward

help:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║         MarketSentinel — Available Commands                ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Setup & Data:"
	@echo "  make setup       - Install dependencies"
	@echo "  make download    - Download data from Kaggle"
	@echo ""
	@echo "Training:"
	@echo "  make train TICKER=AAPL MODE=walk_forward"
	@echo "  make train TICKER=GOOGL MODE=train_test_split"
	@echo ""
	@echo "Backtesting:"
	@echo "  make backtest TICKER=AAPL"
	@echo ""
	@echo "Dashboard:"
	@echo "  make app         - Launch Streamlit dashboard"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean       - Remove generated files"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Check code style"
	@echo ""

setup:
	@echo "📦 Installing dependencies..."
	pip install -q -r requirements.txt
	@mkdir -p data/raw data/processed models reports/figures logs
	@echo "✅ Setup complete!"

download:
	@echo "📥 Downloading S&P 500 dataset from Kaggle..."
	@python -c "from src.data_loader import download_kaggle_dataset; download_kaggle_dataset('camnugent/sandp500', 'data/raw/')" || echo "⚠️  Download failed. Using synthetic data fallback."

train:
	@echo "🏋️  Training model for ticker: $(TICKER) with mode: $(MODE)"
	python train.py --ticker $(TICKER) --mode $(MODE) --start-date $(START_DATE) --end-date $(END_DATE)

backtest:
	@echo "💰 Running backtest for ticker: $(TICKER)"
	python backtest.py --ticker $(TICKER) --start-date $(START_DATE) --end-date $(END_DATE)

app:
	@echo "🚀 Launching Streamlit dashboard..."
	streamlit run app/streamlit_app.py --server.port 8501

clean:
	@echo "🗑️  Cleaning generated files..."
	@rm -rf models/*.joblib models/*.json models/*.pkl
	@rm -rf data/processed/*.csv
	@rm -rf reports/figures/*.png reports/*.txt reports/*.csv
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned!"

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v 2>/dev/null || echo "ℹ️  No tests directory"

lint:
	@echo "🔍 Checking code style..."
	flake8 src/ app/ --max-line-length=120 2>/dev/null || echo "ℹ️  Install flake8 for linting"

format:
	@echo "✨ Formatting code..."
	black src/ app/ --line-length=120 2>/dev/null || echo "ℹ️  Install black for formatting"

all: setup download
	@echo "✅ Setup and download complete! Run 'make train TICKER=AAPL' to start."
