# Project 08 — Topic Modeling (Unsupervised NLP)

> Discover latent themes in the 20 Newsgroups corpus using NMF and LDA, with a
> fully modular pipeline, evaluation metrics, and a Streamlit web application.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Installation](#installation)
6. [Running the Project (Step-by-Step)](#running-the-project-step-by-step)
7. [Streamlit App](#streamlit-app)
8. [Configuration](#configuration)

---

## Overview

Topic modeling is an **unsupervised** NLP technique that discovers abstract "topics" — coherent
clusters of words — from a collection of documents.  This project implements two industry-standard
algorithms:

| Algorithm | Full Name | Best for |
|---|---|---|
| **NMF** | Non-negative Matrix Factorisation | Sparse TF-IDF input, crisp topics |
| **LDA** | Latent Dirichlet Allocation | Count input, probabilistic interpretation |

**Dataset:** [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) — 18,846 documents, 20 categories.

---

## Project Structure

```
project-08-topic-modeling/
│
├── data/
│   ├── raw/                    ← Auto-fetched raw pickle
│   ├── interim/                ← Intermediate representations
│   └── processed/              ← Final cleaned CSV + pickle
│
├── notebooks/
│   ├── 01_eda.ipynb            ← Exploratory data analysis
│   ├── 02_preprocessing.ipynb  ← Cleaning & vectorisation walkthrough
│   └── 03_modeling.ipynb       ← Model training, evaluation, visualisation
│
├── src/
│   ├── data/
│   │   ├── load_data.py        ← Dataset download & persistence
│   │   └── make_dataset.py     ← Full data preparation pipeline (CLI)
│   │
│   ├── features/
│   │   ├── preprocess.py       ← Text cleaning + sklearn transformer
│   │   └── vectorize.py        ← TF-IDF / Count vectorisation
│   │
│   ├── models/
│   │   ├── train.py            ← NMF / LDA training + topic extraction (CLI)
│   │   ├── evaluate.py         ← Reconstruction error, diversity, coherence
│   │   └── predict.py          ← TopicPredictor inference class
│   │
│   ├── visualization/
│   │   └── plots.py            ← All matplotlib figures
│   │
│   └── utils.py                ← Logging, paths, config, seeds
│
├── models/                     ← Saved joblib artefacts
├── reports/
│   ├── figures/                ← PNG outputs
│   ├── topics.csv              ← Top-word table
│   └── report.md               ← Auto-generated insights report
│
├── app/
│   └── app.py                  ← Streamlit web application
│
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Data Acquisition
`src/data/load_data.py` fetches the dataset via `sklearn.datasets.fetch_20newsgroups`.
Email headers, footers, and quoted replies are stripped at fetch time.

### 2. Preprocessing (`src/features/preprocess.py`)
1. Remove email artifacts (addresses, URLs, header lines)
2. Remove punctuation and digits
3. Lowercase
4. Tokenise (NLTK `word_tokenize`)
5. Remove English stop-words + custom 20-NG noise words
6. WordNet lemmatisation
7. Second stop-word pass post-lemmatisation
8. Minimum token length filter (≥ 3 chars)

### 3. Feature Engineering (`src/features/vectorize.py`)
- **TF-IDF** (sublinear TF scaling, L2 norm) — for NMF
- **Count Vectorizer** (raw integer counts) — for LDA
- Maximum vocabulary: 5,000 terms (configurable)

### 4. Modeling (`src/models/train.py`)
Both models are trained via a unified `build_model(cfg)` factory.
Switch algorithms via `config.json` or CLI argument.

### 5. Evaluation (`src/models/evaluate.py`)
| Metric | Model | Interpretation |
|---|---|---|
| Reconstruction Error | NMF | Frobenius ‖V − WH‖; lower = better fit |
| Perplexity | LDA | Per-word log-likelihood bound; lower = better |
| Topic Diversity | Both | Fraction of unique words in top-K; higher = less overlap |
| UMass Coherence | Both | Intrinsic coherence; higher (less negative) = more coherent |

### 6. N-Topics Sweep
`evaluate.sweep_n_topics()` trains models for a configurable range of topic counts and produces
a comparison CSV + plot.

---

## Results

- **Best model:** NMF with n=10 topics.
- Topics are well-separated and align with known thematic clusters
  (space, religion, politics, sports, hardware, software, …).
- See `reports/report.md` for the full findings and `reports/topics.csv` for topic keywords.

---

## Installation

```bash
# 1. Clone repository
git clone https://github.com/yourname/project-08-topic-modeling.git
cd project-08-topic-modeling

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Project (Step-by-Step)

### Step 1 — Download & prepare dataset
```bash
python -m src.data.make_dataset
```
Outputs:
- `data/raw/newsgroups_raw.pkl`
- `data/interim/newsgroups_interim.pkl`
- `data/processed/newsgroups_processed.csv`
- `data/processed/newsgroups_processed.pkl`

---

### Step 2 — Run EDA notebook
```bash
jupyter notebook notebooks/01_eda.ipynb
```
Run all cells. Saves figures to `reports/figures/`.

---

### Step 3 — Run Preprocessing notebook (optional)
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```
Demonstrates the cleaning and vectorisation pipeline interactively.

---

### Step 4 — Train the model
```bash
python -m src.models.train
```
Outputs:
- `models/nmf_model.joblib`
- `models/vectorizer.joblib`
- `reports/topics.csv`

To train LDA instead, edit `config.json` → `"model_type": "lda"` or create one:
```json
{
  "model_type": "lda",
  "n_topics": 10,
  "vectorizer": "count"
}
```
Then:
```bash
python -m src.models.train --config config.json
```

---

### Step 5 — Run Modeling notebook (optional)
```bash
jupyter notebook notebooks/03_modeling.ipynb
```
Runs the full experiment including sweep, heatmap, and word clouds.

---

### Step 6 — Launch the Streamlit app
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.

---

## Streamlit App

The app provides:
- **Predict tab** — Paste any text, see dominant topic + keyword chips + weight bar chart.
- **Browse Topics tab** — Expandable cards for all discovered topics.
- Sidebar model selector (NMF / LDA) and display tuning.

Screenshot: See `reports/figures/app_screenshot.png` (generated after first run).

---

## Configuration

All pipeline parameters are centralised in `src/utils.load_config()`.
Override any value by passing a `config.json`:

```json
{
  "n_topics": 15,
  "model_type": "nmf",
  "max_features": 8000,
  "vectorizer": "tfidf",
  "max_df": 0.95,
  "min_df": 2,
  "n_top_words": 15,
  "random_state": 42,
  "nmf": {
    "init": "nndsvda",
    "max_iter": 400
  },
  "lda": {
    "max_iter": 25,
    "learning_method": "online"
  }
}
```

---

## Tech Stack

| Component | Library |
|---|---|
| Data | scikit-learn (20 Newsgroups) |
| Preprocessing | NLTK |
| Vectorisation | scikit-learn `TfidfVectorizer`, `CountVectorizer` |
| Modeling | scikit-learn `NMF`, `LatentDirichletAllocation` |
| Persistence | joblib |
| Visualisation | matplotlib, wordcloud |
| Web app | Streamlit |
| Notebooks | Jupyter |
