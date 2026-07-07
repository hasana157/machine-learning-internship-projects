.PHONY: setup data train evaluate app clean help

# ── VisualSentry Makefile ─────────────────────────────────────────────────────

PYTHON    := python3
PIP       := pip3
CONFIG    := config.yaml
APP_FILE  := app/streamlit_app.py

help:  ## Show this help message
	@echo ""
	@echo "  VisualSentry — AI-Powered Visual Defect Detection"
	@echo "  ─────────────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""

setup:  ## Install all Python dependencies from requirements.txt
	$(PIP) install -r requirements.txt

data:  ## Generate synthetic demo data (normal + defective images)
	$(PYTHON) -c "from src.data_loader import generate_demo_data; generate_demo_data('$(CONFIG)')"

train:  ## Train the ConvAutoencoder on normal images
	$(PYTHON) train.py --config $(CONFIG)

evaluate:  ## Run evaluation on the full normal + defect set
	$(PYTHON) evaluate.py --config $(CONFIG)

app:  ## Launch the Streamlit GUI
	streamlit run $(APP_FILE) --server.headless true

all: setup data train evaluate app  ## Run the full pipeline end-to-end

clean:  ## Remove generated data, models, and reports
	rm -rf data/normal/* data/defect/*
	rm -rf models/*.h5
	rm -rf reports/figures/*.png reports/*.csv
	@echo "Cleaned generated artefacts."
