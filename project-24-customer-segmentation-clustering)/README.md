# CustomerSegment AI

Customer RFM Clustering & Persona Intelligence Platform.

Transforms raw transaction data into business-meaningful customer segments
using RFM (Recency, Frequency, Monetary) feature engineering, KMeans
clustering with silhouette-based validation, and automated persona
generation — visualized in an interactive Streamlit dashboard, with an
optional real-time FastAPI scoring endpoint.

---

## 1. What's in this project

```
CustomerSegmentAI/
├── app.py                      # Streamlit dashboard (5 tabs)
├── run_pipeline.py              # End-to-end batch pipeline (run this first)
├── requirements.txt
├── README.md                    # <- you are here
├── data/
│   ├── generate_synthetic_data.py   # Synthetic dataset generator (no internet needed)
│   └── raw/transactions.csv         # created after you run the generator/pipeline
├── src/
│   ├── rfm_calculator.py        # RFMCalculator: transactions -> RFM features
│   ├── clustering_engine.py     # ClusteringEngine: KMeans + silhouette optimization
│   └── persona_generator.py     # PersonaGenerator: cluster stats -> business personas
├── api/
│   └── main.py                  # Optional FastAPI real-time scoring service
├── scripts/
│   └── weekly_retrain.py        # Scheduled retraining + drift detection
├── tests/                       # pytest unit tests (28 tests across 3 modules)
├── models/                      # generated artifacts (scaler, KMeans model, personas.json, metrics.json)
└── logs/                        # created by the retraining script
```

---

## 2. Which dataset to use

You have two options. **Both work out of the box.**

### Option A — Ships with the project (recommended to start): Synthetic E-Commerce Data
`data/generate_synthetic_data.py` generates a realistic, reproducible
transaction dataset (~50,000 transactions, 5,000 customers) using only
`numpy`/`pandas` — **no download, no API keys, no internet required**.
Purchase frequency follows a Pareto-style skew (a small share of
customers drive a disproportionate share of revenue), which is what
gives KMeans real structure to find. `run_pipeline.py` will auto-generate
this data the first time you run it if `data/raw/transactions.csv`
doesn't exist yet.

Use this to get the whole system running in under a minute and to see
every tab of the dashboard populated immediately.

### Option B — Real-world data (recommended for a portfolio/production build): UCI Online Retail
- Source: https://archive.ics.uci.edu/dataset/352/online+retail
- ~500,000 transactions, ~5,000 customers, real UK e-commerce data
  (Dec 2010 – Dec 2011)
- Columns: `InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`

**To use it:**
1. Download `Online Retail.xlsx` from the link above.
2. Convert/save it as CSV with exactly those column names, e.g.:
   ```python
   import pandas as pd
   df = pd.read_excel("Online Retail.xlsx")
   df.to_csv("data/raw/transactions.csv", index=False)
   ```
3. Run `python run_pipeline.py` — it will use your file instead of
   generating synthetic data, since `data/raw/transactions.csv` will
   already exist.

The real UCI dataset is right-skewed/Pareto-shaped in a way the
synthetic generator only approximates, so expect somewhat different
(often better-separated) clusters and silhouette scores once you swap
it in — this is expected and is called out in the SRS risk register.

---

## 3. Setup guide

### Prerequisites
- Python 3.10+
- pip

### Step-by-step

```bash
# 1. Unzip and enter the project
unzip CustomerSegmentAI.zip
cd CustomerSegmentAI

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
#    (auto-generates synthetic data on first run, computes RFM,
#     fits KMeans with silhouette optimization, generates personas,
#     and writes all artifacts to models/ and data/)
python run_pipeline.py

# 5. Launch the dashboard
streamlit run app.py
```

Your browser should open automatically to `http://localhost:8501`. If not,
open that URL manually.

### Running the tests

```bash
pytest tests/ -v
```

28 tests cover `RFMCalculator`, `ClusteringEngine`, and `PersonaGenerator`
— feature computation, validation, scaling, cluster fitting/prediction,
persona rules, and save/load round-trips.

### Optional: real-time scoring API

```bash
uvicorn api.main:app --reload --port 8000
```

Then, e.g.:
```bash
curl -X POST http://localhost:8000/predict-cluster \
  -H "Content-Type: application/json" \
  -d '{"recency": 12, "frequency": 90, "monetary": 13000}'
```
returns the assigned cluster, persona name, and recommended action.
Requires `python run_pipeline.py` to have been run at least once
(the API loads the saved scaler + KMeans model from `models/`).

### Optional: weekly retraining job

```bash
python scripts/weekly_retrain.py
```

Recomputes RFM from `data/raw/transactions.csv`, runs a KS-test drift
check against the last saved RFM snapshot, refits KMeans, and updates
`models/` in place. Schedule it with cron, e.g. every Sunday at 23:00:
```
0 23 * * 0 cd /path/to/CustomerSegmentAI && python scripts/weekly_retrain.py >> logs/retrain.log 2>&1
```

---

## 4. How the pipeline works

1. **RFM feature engineering** (`src/rfm_calculator.py`): cleans
   transactions (drops returns/nulls), computes Recency (days since last
   purchase), Frequency (distinct invoice count in the last 365 days),
   and Monetary (sum of Quantity × UnitPrice), then fits a
   `StandardScaler` on the three features.
2. **KMeans + silhouette optimization** (`src/clustering_engine.py`):
   grid-searches `k` in **3–5** (the business-required persona-tier
   range), picks the `k` with the best silhouette score (preferring the
   smaller `k` on a plateau), refits with a higher `n_init` for
   stability, and reports Davies-Bouldin and Calinski-Harabasz as
   supplementary validation metrics.
3. **Persona generation** (`src/persona_generator.py`): ranks each
   cluster's mean Recency/Frequency/Monetary against the population's
   33rd/66th percentiles, assigns a persona name (Platinum / Gold /
   Silver / At-Risk), and attaches a plain-English insight and a list of
   recommended marketing actions.
4. **Dashboard** (`app.py`): loads the saved artifacts and renders 5
   tabs — Overview, Profiles, Member Explorer, Silhouette Analysis, and
   Segmentation Health.

## 5. Interpreting the silhouette score

| Score | Meaning |
|---|---|
| > 0.50 | Well-separated, highly actionable segments |
| 0.25 – 0.50 | Overlapping but still interpretable clusters |
| < 0.25 | Weak segmentation — try different features (e.g. log-transform Frequency/Monetary), a different `k` range, or more data |

On the bundled synthetic dataset you should see a silhouette score
around **0.35–0.45** for k=3, which is realistic for demo data. The real
UCI Online Retail dataset (Option B above) more closely matches the
target of ≥ 0.50 referenced in the platform's original spec, because its
customer behavior is more sharply bimodal than the synthetic generator's
smoother distribution.

## 6. Troubleshooting

- **"No pipeline artifacts found" in the dashboard** → run
  `python run_pipeline.py` first; `app.py` only reads from `models/` and
  `data/`, it never recomputes on its own.
- **`ModuleNotFoundError: No module named 'pyarrow'`** → harmless; the
  pipeline automatically falls back to CSV instead of Parquet.
- **Silhouette warning at startup** → informational only; it fires when
  the best silhouette in the searched `k` range is below 0.30, so you
  know to investigate feature quality before trusting the segments.
- **Streamlit port already in use** → `streamlit run app.py --server.port 8502`.
