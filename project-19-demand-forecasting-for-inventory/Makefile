.PHONY: setup train forecast app clean help

help:
	@echo "ForecastIQ — Demand Forecasting for Retail Inventory"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup       Install dependencies"
	@echo "  make train       Train the demand forecaster"
	@echo "  make forecast    Generate a 30-day forecast for store 1"
	@echo "  make app         Launch Streamlit dashboard"
	@echo "  make clean       Remove generated artifacts"

setup:
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

train:
	python train.py

forecast:
	python forecast.py --store 1 --days 30

app:
	streamlit run app/streamlit_app.py --server.port 8501

clean:
	rm -rf models/*.joblib
	rm -rf reports/figures/*.png
	rm -rf reports/*.csv
	rm -rf reports/*.txt
	@echo "✅ Cleaned up generated files"
