# Makefile
# --------
# Common project tasks. Run with `make <target>`.

.PHONY: install eda train-all train-baseline train-transfer test app clean

install:
	pip install -r requirements.txt

eda:
	python notebooks/01_eda.py

train-baseline:
	python train.py --mode baseline

train-transfer:
	python train.py --mode transfer

train-all:
	python train.py --mode both

test:
	pytest tests/ -v --tb=short

app:
	streamlit run app/streamlit_app.py

compare:
	python notebooks/02_model_comparison.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

help:
	@echo ""
	@echo "  make install          Install dependencies"
	@echo "  make eda              Run EDA script (saves plots to reports/figures/)"
	@echo "  make train-all        Train both baseline + transfer models"
	@echo "  make train-baseline   Train baseline CNN only"
	@echo "  make train-transfer   Train MobileNetV2 transfer model only"
	@echo "  make test             Run unit tests"
	@echo "  make app              Launch Streamlit app"
	@echo "  make compare          Run model comparison notebook"
	@echo "  make clean            Remove Python cache files"
	@echo ""
