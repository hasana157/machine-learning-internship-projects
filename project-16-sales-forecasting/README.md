# Project 16 — Sales Forecasting

A time-series forecasting system that engineers lag/rolling features,
trains a linear baseline against a Random Forest, evaluates both with
proper chronological validation, and ships an interactive Streamlit
dashboard on top.

```
Actual sales  ──▶  feature engineering  ──▶  Linear Regression  ──┐
                    (lags, rolling stats,                          ├──▶ MAE / RMSE / MAPE
                     calendar features)   ──▶  Random Forest      ──┘     + forecast plots
```

---

## 1. Project structure

```
project-16-sales-forecasting/
├── app/
│   └── streamlit_app.py       # interactive dashboard
├── data/
│   └── sales_data.csv         # dataset lives here (generated or your own)
├── models/                    # saved model + feature-name artifacts (generated)
├── reports/
│   ├── evaluation_report.txt  # metrics summary (generated)
│   └── figures/               # forecast / residual / importance plots (generated)
├── src/
│   ├── data_generator.py      # synthetic dataset generator
│   ├── features.py            # lag + rolling-stat + calendar features
│   ├── model.py                # model factory (linear / random forest)
│   ├── trainer.py             # data → features → split → fit → save
│   ├── evaluator.py           # metrics + plots
│   └── utils.py               # config loader, logger, dir setup
├── config.yaml                # single source of truth for every parameter
├── train.py                   # CLI entrypoint
├── Makefile                   # setup / data / train / evaluate / app / clean
├── requirements.txt
└── README.md
```

---

## 2. Setup

```bash
git clone <YOUR_REPO_URL> project-16-sales-forecasting
cd project-16-sales-forecasting

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# or: make setup
```

---

## 3. Dataset — where it comes from and where it goes

You have two options. Either works with **zero code changes** — the
pipeline always looks for `data/sales_data.csv` first, and only
generates a synthetic dataset if that file is missing.

### Option A — Synthetic data (default, no download needed)

Nothing to do. The first time you run training or the app, `src/data_generator.py`
creates ~2 years of realistic daily sales (trend + weekly/yearly seasonality
+ promo days + holiday spikes + noise) and writes it to `data/sales_data.csv`.

To (re)generate it explicitly:

```bash
make data
# or
python -c "from src.utils import load_config; from src.data_generator import generate_sales_data; c=load_config(); generate_sales_data(c).to_csv(c['paths']['data'], index=False)"
```

Tune the shape of the data (trend strength, noise, promo frequency, etc.)
in `config.yaml` under the `data:` key.

### Option B — A real dataset

Drop any of these into `data/sales_data.csv` and it will just work:

| Source | Where to get it |
|---|---|
| Rossmann Store Sales | https://www.kaggle.com/competitions/rossmann-store-sales |
| Store Item Demand Forecasting Challenge | https://www.kaggle.com/competitions/demand-forecasting-kernels-only |
| Walmart Recruiting — Store Sales Forecasting | https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting |
| Superstore Sales (sample dataset) | https://www.kaggle.com/datasets/vivek468/superstore-dataset-final |

**Required schema** — the pipeline only needs two columns (extra columns
are ignored):

| column | type | description |
|---|---|---|
| `date`  | `YYYY-MM-DD` | one row per day, sorted or not (the code sorts it) |
| `sales` | number | the value you want to forecast |

If your raw file has multiple stores/products, aggregate to a single daily
total first, e.g.:

```python
import pandas as pd
raw = pd.read_csv("raw_rossmann.csv", parse_dates=["Date"])
daily = raw.groupby("Date")["Sales"].sum().reset_index()
daily.columns = ["date", "sales"]
daily.to_csv("data/sales_data.csv", index=False)
```

Then just run training normally — `is_promo` / `is_holiday` columns are
optional and only used by the synthetic generator; the feature pipeline
doesn't require them.

---

## 4. Running the project

### Train a single model

```bash
python train.py --model_type linear
python train.py --model_type rf
```

### Train and compare both (recommended)

```bash
python train.py --model_type both
# or: make train
```

This will:
1. Load `data/sales_data.csv` (generating it if missing)
2. Build lag features (1/7/14/28-day), rolling mean/std (7/14/30-day), and calendar features
3. Split **chronologically** (80/20 by default — no shuffling, no leakage)
4. Fit each model, print MAE / RMSE / MAPE
5. Save models to `models/`
6. Save forecast, residual, and feature-importance plots to `reports/figures/`
7. Write `reports/evaluation_report.txt`

### Re-evaluate saved models without retraining

```bash
python -m src.evaluator
# or: make evaluate
```

### Launch the interactive dashboard

```bash
streamlit run app/streamlit_app.py
# or: make app
```

Open http://localhost:8501 to:
- switch between synthetic data and your own uploaded CSV
- pick Linear Regression or Random Forest
- adjust the train/test split live
- view the forecast chart, feature importances, and metrics
- download the forecast as CSV

---

## 5. Feature engineering

| Feature | Why it matters |
|---|---|
| `lag_1, lag_7, lag_14, lag_28` | yesterday / same-day-last-week / etc. — strong autocorrelation signal |
| `roll_mean_7/14/30`, `roll_std_7/14/30` | smoothed recent level and recent volatility |
| `day_of_week`, `is_weekend` | captures weekly seasonality |
| `month`, `day_of_year` | captures yearly seasonality |

All rolling/lag windows are **shifted by one day before** aggregating,
so no feature for day *t* ever uses information from day *t* or later.

---

## 6. Models

- **Baseline:** Linear Regression — fast, interpretable, a fair floor to beat.
- **Improved:** Random Forest Regressor — captures non-linear seasonality
  and interactions between lag/rolling features that a linear model can't.

## 7. Evaluation

Metrics computed on the held-out chronological test window:
- **MAE** — average absolute forecast error, in sales units
- **RMSE** — penalizes large misses more heavily than MAE
- **MAPE (%)** — error as a percentage of actual sales, easy to communicate

## Key insight

Time-series problems require respecting temporal order end-to-end — from
the train/test split down to how rolling features are computed. Random
shuffling, or forgetting to shift a rolling window by one day, silently
leaks the future into the training signal and makes every metric a lie.

---

## 8. Deliverables generated after a full run

```
models/linear_model.joblib
models/rf_model.joblib
models/linear_feature_names.json
models/rf_feature_names.json
reports/evaluation_report.txt
reports/figures/linear_forecast.png
reports/figures/rf_forecast.png
reports/figures/linear_residuals.png
reports/figures/rf_residuals.png
reports/figures/rf_feature_importance.png
```

---

## 9. Push to GitHub

```bash
git init
git add .
git commit -m "Project 16: Sales Forecasting with lag features, rolling stats, and a Streamlit dashboard"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```
