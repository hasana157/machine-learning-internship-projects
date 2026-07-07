# Setup & Run Guide

## 1. Requirements
- Python 3.10+ (3.11/3.13 also fine)
- pip

## 2. Install

```bash
cd project-17-energy-forecasting
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or, if you have `make` installed:
```bash
make setup
```

## 3. Get the dataset — you don't have to

The repo ships with a **synthetic** dataset generator, so there is nothing
to download to try the project. The first time you run training, if
`data/energy_consumption.csv` doesn't exist, it's generated automatically
from `config.yaml` settings.

If you want to use **real** data instead (recommended once you've verified
the pipeline works):

1. Download a daily/hourly energy dataset, e.g. from Kaggle:
   - "Hourly Energy Consumption (PJM Interconnection)"
   - "Household Power Consumption"
2. If your data is hourly, resample to daily totals:
   ```python
   import pandas as pd
   df = pd.read_csv("your_raw_file.csv", parse_dates=["datetime_col"])
   daily = df.resample("D", on="datetime_col")["usage_col"].sum().reset_index()
   daily.columns = ["date", "consumption"]
   daily.to_csv("data/energy_consumption.csv", index=False)
   ```
3. Make sure the final CSV has exactly two columns: `date`, `consumption`.
4. Put it at `data/energy_consumption.csv`, overwriting the synthetic one.

## 4. Train

```bash
python train.py
```
or
```bash
make train
```

This will:
- Load `data/energy_consumption.csv` (generating it first if missing)
- Engineer lag/rolling/calendar features
- Train Linear Regression, Random Forest, and Gradient Boosting
- Save the best model to `models/best_model.joblib`
- Write `reports/evaluation_report.txt`
- Save 3 charts to `reports/figures/`:
  - `model_comparison.png`
  - `forecast_vs_actual.png`
  - `error_by_weekday.png`

## 5. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```
or
```bash
make app
```

Open `http://localhost:8501`. From the sidebar you can upload your own CSV
(same `date`, `consumption` format) and hit **Retrain** to see live results
on your data.

## 6. Push to GitHub

```bash
git init
git add .
git commit -m "Project 17: Energy Consumption Forecasting with Seasonal Analysis"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run commands from the project root, not from inside `src/` or `app/` |
| Streamlit port already in use | `streamlit run app/streamlit_app.py --server.port 8502` |
| Want to reset everything | `make clean` deletes trained models, metadata, and figures (data/*.csv is kept) |
