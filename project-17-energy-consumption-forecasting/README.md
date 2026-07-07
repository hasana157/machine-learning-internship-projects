# LoadCast — Seasonal Energy Consumption Forecasting

<div align="center">

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

### *"Know tomorrow's load, today."*

**LoadCast** is a seasonality-aware daily energy forecasting system that trains
and compares multiple regression models, then breaks down forecast error by
day of week to expose where the model is weakest.

</div>

---

## 📌 Overview

Energy consumption follows patterns at several time scales at once — a slow
long-term drift, a weekly weekday/weekend cycle, and a yearly summer/winter
swing. LoadCast engineers features for all three, trains **Linear Regression**,
**Random Forest**, and **Gradient Boosting** side by side, and automatically
keeps the best performer.

- **Multi-model comparison** — no single algorithm assumed to be best
- **Weekday error breakdown** — surfaces systematic weaknesses (e.g. weekends)
- **Interactive Streamlit dashboard** — upload your own CSV, retrain in one click
- **Fully configurable** via `config.yaml` — no hardcoded parameters

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 📅 **Calendar Features** | day-of-week, day-of-month, month, weekend flag |
| ⏱ **Lag Features** | configurable lags (default: 1, 7, 14 days) |
| 📈 **Rolling Statistics** | rolling mean/std over configurable windows |
| 🤖 **Multi-Model Training** | Linear Regression, Random Forest, Gradient Boosting |
| 🏆 **Automatic Model Selection** | lowest test MAE is saved as production model |
| 🗓️ **Weekday Error Analysis** | bar chart + report of MAE per day of week |
| 📊 **Interactive Dashboard** | Streamlit + Plotly, upload-your-own-CSV support |

---

## 🧠 How It Works

```text
╔════════════════╗    ╔══════════════════════╗    ╔═══════════════════════╗    ╔══════════════════╗
║  Daily Load    ║───▶║  Feature Engineering ║───▶║  3 Candidate Models   ║───▶║  Best Model +     ║
║  Time Series   ║    ║                      ║    ║                       ║    ║  Error Analysis   ║
╚════════════════╝    ╚══════════════════════╝    ╚═══════════════════════╝    ╚══════════════════╝
  Trend + weekly         Lags (1,7,14 days)         Linear / RandomForest /       Lowest test MAE
  + yearly cycles         Rolling mean/std            GradientBoosting              wins → saved
  Synthetic or            Calendar features           trained on 80% split         Weekday error
  uploaded CSV                                        evaluated on last 20%        chart generated
```

### Why compare multiple models instead of picking one?

- **Linear Regression** is a fast, interpretable baseline — if a complex model
  can't beat it, the extra complexity isn't earning its keep.
- **Random Forest** captures non-linear interactions between lag/rolling
  features without needing feature scaling.
- **Gradient Boosting** typically wins on smooth seasonal signals like this
  one, but not always — the comparison step exists precisely so you don't
  have to assume.

---

## 📊 Results Benchmark

Evaluated on the bundled 2-year synthetic dataset (chronological 80/20 split):

| Model | MAE (kWh) | RMSE (kWh) | MAPE |
|---|---|---|---|
| Linear Regression | 19.89 | 24.80 | 4.00% |
| Random Forest | 20.67 | 25.59 | 4.17% |
| **Gradient Boosting (selected)** | **19.81** | **24.68** | **3.98%** |

> ⚠️ Results are for the synthetic default dataset. Run `make train` after
> swapping in your own data to get dataset-specific numbers.

**Weekday error breakdown (best model):** error peaks mid-week in the
synthetic data — with a real dataset this chart is what tells you whether
your model is systematically missing weekend behaviour, billing-cycle
effects, or public holidays.

---

## 🚀 Quick Start

> Get LoadCast running in under 2 minutes with 4 commands.

```bash
# 1. Clone the repository
git clone https://github.com/hasana157/LoadCast && cd LoadCast

# 2. Install dependencies
make setup

# 3. Generate data (synthetic default) and train all models
make train

# 4. Launch the Streamlit dashboard
make app
```

The dashboard opens at `http://localhost:8501`.

---

## 🗄️ Dataset

### Default: synthetic data (works out of the box)
`make train` auto-generates `data/energy_consumption.csv` if it doesn't
already exist, using `src/data_generator.py`. No download required — this is
enough to demo the full pipeline immediately.

### Using a real dataset
LoadCast works with any daily energy CSV. Good free sources:

- Kaggle — [Hourly Energy Consumption (PJM Interconnection)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- Kaggle — [Household Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)
- UCI Machine Learning Repository — Individual Household Electric Power Consumption

**Setup steps:**
1. Download a dataset and, if it's hourly, resample it to daily totals
   (`df.resample("D").sum()` in pandas).
2. Make sure it has exactly two columns: `date` and `consumption`.
3. Save it as `data/energy_consumption.csv`, replacing the synthetic file.
4. Run `make train` (or hit **Retrain** in the dashboard sidebar).

```
date,consumption
2022-01-01,512.3
2022-01-02,498.7
...
```

---

## 📁 Project Structure

```
LoadCast/
│
├── app/
│   └── streamlit_app.py       # Streamlit dashboard entry point
│
├── data/
│   └── energy_consumption.csv # Default synthetic dataset (replaceable)
│
├── models/
│   ├── best_model.joblib      # Saved best-performing model
│   └── model_metadata.json    # Which model won + all model metrics
│
├── reports/
│   ├── evaluation_report.txt  # Text summary: metrics + weekday errors
│   └── figures/
│       ├── error_by_weekday.png
│       ├── forecast_vs_actual.png
│       └── model_comparison.png
│
├── src/
│   ├── data_generator.py      # Synthetic daily load generator
│   ├── features.py            # Calendar, lag, rolling features
│   ├── models.py               # Candidate model registry
│   ├── trainer.py              # Train/evaluate/select-best pipeline
│   └── evaluator.py            # Charts + text report generation
│
├── config.yaml                 # All parameters (no hardcoding)
├── train.py                    # CLI training entry point
├── Makefile                     # One-command setup, train, run
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration (`config.yaml`)

All parameters are centralized — no digging through source files:

```yaml
data:
  n_days: 730
  base_load: 500
  trend_amplitude: 50
  weekly_amplitude: 40
  yearly_amplitude: 60
  noise_sigma: 20

features:
  lag_steps: [1, 7, 14]
  rolling_windows: [7, 14]

model:
  train_split: 0.80
  candidates:
    linear_regression: {}
    random_forest: { n_estimators: 300, max_depth: 12 }
    gradient_boosting: { n_estimators: 300, learning_rate: 0.05, max_depth: 3 }
```

---

## 💡 Use Cases

- 🏠 **Household energy budgeting** — anticipate tomorrow's usage and cost
- 🏢 **Facilities management** — flag days where actual load deviates from forecast
- ⚡ **Utility demand planning** — feed short-horizon forecasts into grid load balancing
- 📉 **Anomaly triage** — large forecast errors on non-weekend days can flag equipment issues

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Machine Learning | scikit-learn (Linear Regression, Random Forest, Gradient Boosting) |
| Data Processing | pandas, numpy |
| Web Application | Streamlit |
| Visualization | Plotly, Matplotlib |
| Configuration | PyYAML |

---

## 🗺️ Roadmap

- [ ] Add SARIMA / Prophet as additional candidate models
- [ ] Public holiday calendar feature
- [ ] Multi-step-ahead forecasting (7/14/30-day horizon)
- [ ] Dockerfile for one-command deployment
- [ ] Confidence intervals on forecasts

---

## 👩‍💻 Author

**Hasana Zahid**
AI & ML Engineer | Python Developer

[![GitHub](https://img.shields.io/badge/GitHub-hasana157-black?logo=github)](https://github.com/hasana157)

---

<div align="center">

⭐ **If LoadCast helped you, consider starring the repo!** ⭐

</div>
