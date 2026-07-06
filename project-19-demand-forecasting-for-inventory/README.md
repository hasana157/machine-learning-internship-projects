# ForecastIQ — AI-Powered Demand Forecasting for Retail Inventory

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/competitions/rossmann-store-sales)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](#)

**Predict demand. Prevent stockouts. Maximize margins.**

ForecastIQ is a **production-grade MLOps pipeline** for demand forecasting trained on the **Rossmann Store Sales Kaggle dataset** — 1,017 German drugstores, 2.5 years of daily sales data, and 1M+ records. It features a **global Random Forest model** with 28+ engineered features, recursive multi-step forecasting, scenario simulation, and an interactive **Streamlit dashboard** for inventory planning.

---

## 🎯 Key Features

- 🏪 **Real Kaggle Data**: 1,017 Rossmann stores · 1M+ rows · Jan 2013 – Jul 2015
- 📊 **RMSPE Metric**: Same evaluation as the official Kaggle competition
- 📅 **28+ Features**: Lags, rolling stats, calendar effects, holiday proximity, store metadata, promotions
- 🔮 **Recursive Forecasting**: Up to 90-day future predictions with ±12% confidence intervals
- 🛍️ **Scenario Simulation**: Model 3 promotion strategies (no promo, weekly, aggressive)
- 🔄 **Auto-Fallback**: Runs on synthetic data if Kaggle CSV unavailable — zero setup friction
- ⚙️ **Fully Configurable**: config.yaml controls all hyperparameters — zero hardcoded magic
- 📊 **Interactive Dashboard**: 5-page Streamlit app with real-time training, store analysis, and error diagnostics
- 🎨 **Publication Quality**: 6 polished evaluation figures + formatted ASCII report

---

## 📦 Project Structure

```
ForecastIQ/
├── app/
│   └── streamlit_app.py              # Multi-page Streamlit dashboard
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Kaggle CSV loader + synthetic fallback
│   ├── features.py                   # 28+ feature engineering pipeline
│   ├── model.py                      # DemandForecaster class (RF + baseline)
│   ├── trainer.py                    # Training orchestration
│   ├── forecaster.py                 # Recursive forecasting + scenarios
│   ├── evaluator.py                  # Metrics, error analysis, figures
│   └── utils.py                      # Config loader, logging, helpers
├── data/
│   ├── train.csv                     # [User adds from Kaggle]
│   ├── store.csv                     # [User adds from Kaggle]
│   └── test.csv                      # [Optional, for submissions]
├── models/
│   └── demand_forecaster.joblib      # Trained model (auto-generated)
├── reports/
│   ├── test_predictions.csv          # Per-row predictions + errors
│   ├── store_metrics.csv             # Per-store performance breakdown
│   ├── evaluation_report.txt         # Formatted ASCII report
│   └── figures/                      # 6 high-quality evaluation charts
│       ├── forecast_sample_grid.png
│       ├── store_error_distribution.png
│       ├── feature_importance.png
│       ├── model_comparison.png
│       ├── sales_by_storetype.png
│       └── evaluation_report.txt
├── .streamlit/
│   └── config.toml                   # Streamlit theme (dark mode, amber accent)
├── config.yaml                       # All configuration (NO hardcoded values)
├── train.py                          # CLI: python train.py
├── forecast.py                       # CLI: python forecast.py --store 1 --days 30
├── requirements.txt
├── Makefile                          # Convenient commands
└── README.md                         # This file
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd ForecastIQ
make setup
```

### 2. (Optional) Download Kaggle Data

**Without this, the system runs in demo mode on synthetic data.**

Download from [kaggle.com/competitions/rossmann-store-sales](https://kaggle.com/competitions/rossmann-store-sales):

```bash
# Save these 3 files to the data/ folder:
data/train.csv          # 1,017,209 rows of daily store sales
data/store.csv          # 1,115 rows of store metadata
data/test.csv           # (Optional, for submissions)
```

### 3. Train the Model

```bash
make train
# Or: python train.py
```

**Output:**
- ✅ Trained model: `models/demand_forecaster.joblib`
- 📊 Evaluation figures: `reports/figures/`
- 📈 Detailed metrics: `reports/evaluation_report.txt`

### 4. Generate Forecasts

```bash
# Single store, 30 days, no promotion
python forecast.py --store 1 --days 30

# With promotion scenarios
python forecast.py --store 5 --days 60 --scenario
```

### 5. Launch Dashboard

```bash
make app
# Or: streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`

---

## 📊 Rossmann Dataset Guide

### Files to Download

From [kaggle.com/competitions/rossmann-store-sales/data](https://kaggle.com/competitions/rossmann-store-sales/data):

| File | Rows | Purpose |
|------|------|---------|
| `train.csv` | 1,017,209 | Daily sales (YYYY-MM-DD format) |
| `store.csv` | 1,115 | Store metadata (type, assortment, competition) |
| `test.csv` | 41,088 | (Optional) Kaggle submission format |

### Key Columns

**train.csv:**
- `Store` → store ID (1–1,115)
- `Date` → date (YYYY-MM-DD string)
- `Sales` → daily sales in euros (TARGET)
- `Customers` → customer count
- `Open` → 0/1 (store open)
- `Promo` → 0/1 (promotion running)
- `DayOfWeek` → 1=Mon…7=Sun
- `StateHoliday` → 0=none, a=public, b=Easter, c=Christmas
- `SchoolHoliday` → 0/1

**store.csv:**
- `Store` → store ID
- `StoreType` → a/b/c/d
- `Assortment` → a/b/c
- `CompetitionDistance` → meters to nearest competitor
- `Promo2` → 0/1 (ongoing promo)
- `PromoInterval` → months when Promo2 is active

---

## 🏗️ Architecture Overview

### Data Flow

```
Kaggle CSV Files (or synthetic fallback)
          ↓
    Data Loader
    - Clean & validate
    - Handle missing values
    - Filter (Open=1, Sales>0)
          ↓
  Feature Engineering (28+ features per-store)
    - Calendar (day, week, month, holidays)
    - Lag features (1, 7, 14, 28, 365 days)
    - Rolling stats (mean, std, min, max, EWM)
    - Velocity & acceleration
    - Promo features
          ↓
  Train/Test Split (time-series aware, 85/15)
          ↓
  Model Training
    - Random Forest (300 trees, global across all stores)
    - Linear Regression (baseline)
    - OOB Score tracking
          ↓
  Evaluation
    - MAE, RMSE, MAPE, RMSPE (Kaggle), R²
    - Per-store breakdown
    - Error analysis & correlations
    - 6 publication-quality figures
          ↓
  Forecasting
    - Recursive auto-regressive prediction
    - Confidence intervals (±12%)
    - Scenario simulation (3 promo strategies)
          ↓
  Interactive Dashboard (Streamlit)
    - 5 pages (Overview, Forecast, Error Analysis, Train, About)
    - Real-time model training
    - Scenario comparison
    - Store-level diagnostics
```

---

## 📈 Model Performance

After training on real Kaggle data:

| Metric | Random Forest | Linear Baseline | Improvement |
|--------|---------------|-----------------|------------|
| **RMSPE** | ~0.15–0.20 | ~0.25–0.30 | **40–50% better** |
| **MAE** | ~300–400 | ~500–600 | Better |
| **MAPE** | ~12–15% | ~18–22% | Better |
| **R²** | ~0.85–0.90 | ~0.70–0.75 | Better |

**Note:** Kaggle competition winning solutions achieved RMSPE ≈ 0.10 with ensemble methods. ForecastIQ's RMSPE ≈ 0.15–0.20 is competitive for a clean, interpretable single-model pipeline.

---

## 🎯 Use Cases

| Scenario | How ForecastIQ Helps |
|----------|---------------------|
| **Inventory Planning** | Predict 30–90 day demand by store type → optimize stock levels |
| **Staffing** | Forecast sales volume → schedule accordingly |
| **Promotion Analysis** | Simulate 3 scenarios → measure expected uplift |
| **Store Performance** | Error analysis shows which stores are hardest to forecast → focus improvements there |
| **Demand Anomalies** | Residual analysis identifies unusual days → investigate causes |
| **Supplier Planning** | Forward-looking predictions → improve procurement |

---

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
data:
  kaggle_train: data/train.csv
  kaggle_store: data/store.csv
  synthetic_stores: 20           # Demo mode store count

features:
  lag_steps: [1, 7, 14, 28, 365]
  rolling_windows: [7, 14, 28]
  ewm_spans: [7, 28]

model:
  n_estimators: 300              # Random Forest trees
  max_depth: null                # Unlimited (or set to 15)
  min_samples_leaf: 2

forecaster:
  horizon_days: 30               # Default forecast window
  confidence_interval: 0.12      # ±12% bands

evaluation:
  train_split_ratio: 0.85        # 85% train, 15% test
  top_errors_n: 15               # Top N hardest stores to show
```

**All values are read from YAML — no hardcoding in Python code.**

---

## 📊 Dashboard Pages

### 🏠 Overview
- 5 metric cards (stores, records, date range, avg sales, best type)
- Daily sales trend (line chart + 28-day MA)
- Sales distribution by store type (box plot)
- Day-of-week × month seasonality heatmap
- Promotion impact analysis
- Recent data table

### 🔮 Forecast
- Store selector with type label
- Horizon slider (7–90 days)
- Promo date selector
- Scenario mode toggle (3 strategies)
- Interactive forecast chart with CI bands
- Auto-generated explanation
- Downloadable CSV

### 📊 Error Analysis
- Store type × assortment error heatmap
- Top 15 hardest-to-forecast stores
- Sales volatility vs error correlation
- Top 20 feature importances
- Full model comparison table
- Evaluation report

### ⚙️ Train Model
- Hyperparameter sliders
- Real-time training progress bar
- Metric cards (RMSPE, MAE, MAPE, R²)
- Model save/load status

### ℹ️ About
- Project overview
- Architecture diagram
- Tech stack
- Kaggle context & performance notes
- FAQ on global vs per-store models
- Author info

---

## 💻 API / Programmatic Use

### Training

```python
from src.trainer import train_demand_forecaster

model, metrics = train_demand_forecaster("config.yaml")
print(f"RMSPE: {metrics['rf_rmspe']:.4f}")
```

### Forecasting

```python
from src.forecaster import forecast_future
from src.model import DemandForecaster

model = DemandForecaster.load("models/demand_forecaster.joblib")
df_train = load_data(config)
df_feat = engineer_features(df_train, config)

# Predict next 30 days for store 1 (no promo)
forecast_df = forecast_future(
    model=model,
    df=df_feat,
    store_id=1,
    horizon_days=30,
    promo_schedule=[0] * 30,
    config=config
)
print(forecast_df)
```

### Scenario Simulation

```python
from src.forecaster import scenario_simulation

scenarios = scenario_simulation(
    model=model,
    df=df_feat,
    store_id=5,
    horizon_days=60,
    config=config
)

for scenario_name, forecast_df in scenarios.items():
    print(f"{scenario_name}: €{forecast_df['forecasted_sales'].sum():,.0f} total")
```

---

## 🔬 Feature Engineering Details

### Calendar Features (from date)
- `day_of_week`, `day_of_month`, `week_of_year`, `month`, `quarter`, `year`
- `is_weekend`, `is_month_start`, `is_month_end`
- `days_to_christmas`, `days_to_easter` (proximity, capped)

### Lag Features (per-store, 1/7/14/28/365 days)
- `lag_1`, `lag_7`, `lag_14`, `lag_28`, `lag_365`
- Captures historical demand patterns
- lag_365 = same weekday last year (critical for seasonal adjustment)

### Rolling Statistics (7/14/28-day windows)
- `roll_mean_7`, `roll_std_7`, `roll_min_7`, `roll_max_7`
- `roll_mean_14`, `roll_median_14`
- `roll_ewm_7`, `roll_ewm_28` (exponential weighted moving average)

### Derived Features
- `sales_velocity_7` = lag_1 - lag_7 (recent trend)
- `sales_accel` = lag_1 - 2×lag_7 + lag_14 (acceleration)
- `cv_7` = roll_std_7 / roll_mean_7 (coefficient of variation)

### Promotional Features
- `promo` (binary, same day)
- `promo_lag_1` (was there promo yesterday?)
- `promo_roll_7` (promos in last 7 days)

### Store Metadata
- `store_type` (a/b/c/d, one-hot encoded)
- `assortment` (a/b/c, one-hot encoded)
- `competition_distance` (numeric, meters)
- `state_holiday` (numeric: 0=none, 1=public, 2=Easter, 3=Christmas)

**⚠️ All lag and rolling features computed PER STORE GROUP to avoid data leakage.**

---

## 🎓 Why This Approach?

### Global Model (Not Per-Store)
- **Cross-store transfer learning**: Small stores benefit from patterns in large stores
- **Statistical power**: 1M+ rows → better feature importances than per-store ARIMA
- **Shared seasonality**: Weekly/monthly patterns apply across all stores
- **Scalability**: One model for 1,000 stores, not 1,000 models

### Random Forest (Not Deep Learning)
- **Interpretability**: Feature importances are clear and trustworthy
- **No hyperparameter tuning pain**: RF is robust out-of-the-box
- **Low computational cost**: Trains in seconds, not hours
- **Production stable**: No OOM issues, deterministic outputs

### RMSPE Metric
- **Kaggle standard**: Directly comparable to competition results
- **Percentage-based**: Treats 10€ errors the same regardless of absolute sales volume
- **Portfolio credibility**: Mentioning RMSPE score shows you know this competition

---

## 📝 Code Quality Standards

✅ **Type hints** on every function  
✅ **Google-style docstrings** on every module & function  
✅ **Zero hardcoding** — all params from config.yaml  
✅ **Logging** — not print() statements  
✅ **Interactive charts** — Plotly (not static Matplotlib)  
✅ **Caching** — @st.cache_data / @st.cache_resource  
✅ **Error handling** — graceful degradation  
✅ **RMSPE display** — Kaggle metric everywhere  

---

## 🐛 Troubleshooting

### Error: "Model not found"
```
Run: python train.py
```

### Error: "Kaggle data not found"
The system auto-falls back to synthetic data. To use real data:
1. Download train.csv and store.csv from Kaggle
2. Place in `data/` folder
3. Re-run `python train.py`

### Slow training
- Reduce `n_estimators` in config.yaml (default: 300)
- Use `max_depth: 15` instead of null
- Ensure you have 4+ CPU cores available

### Forecast values are too conservative
- Check `confidence_interval` in config.yaml (default: 0.12 = ±12%)
- Review feature importances — maybe a key feature is missing

---

## 📚 References

- **Kaggle Competition**: [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)
- **scikit-learn**: [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Plotly**: [plotly.com/python](https://plotly.com/python/)

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Hasana Zahid**

- 🔗 [GitHub](https://github.com/hasana157)
- 💼 [LinkedIn](https://linkedin.com/in/hasana-zahid)
- 📧 [Email](mailto:your-email@example.com)

---

## 🙏 Acknowledgments

- **Kaggle** for the Rossmann Store Sales dataset and competition
- **scikit-learn** for the excellent ML library
- **Streamlit** for making interactive dashboards accessible
- **The data science community** for best practices in MLOps

---

## 🎯 Next Steps

1. **Train on full Kaggle data** (1M+ rows) for best performance
2. **Tune hyperparameters** → `GridSearchCV` for optimal RMSPE
3. **Ensemble models** → Stack RF + XGBoost + LightGBM for competition-grade results
4. **Real-time serving** → Deploy with FastAPI + Docker
5. **A/B testing** → Measure forecast accuracy in production

---

**Made with ❤️ for retail demand forecasting excellence.**
