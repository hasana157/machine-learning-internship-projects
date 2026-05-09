# Project 10 — Resume Screening Model

> **⚠️ IMPORTANT — Educational Use Only**
> This project is strictly for learning NLP / ML classification.
> It must **NOT** be used for real hiring, candidate rejection, or recruitment automation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Dataset & Limitations](#3-dataset--limitations)
4. [Installation](#4-installation)
5. [How to Run (Step-by-Step)](#5-how-to-run-step-by-step)
6. [Pipeline Architecture](#6-pipeline-architecture)
7. [Model Performance & Honest Reporting](#7-model-performance--honest-reporting)
8. [Responsible AI — Ethical Limitations](#8-responsible-ai--ethical-limitations)
9. [Tech Stack](#9-tech-stack)

---

## 1. Project Overview

A production-quality multi-class NLP classification system that categorises
resume text into one of five job roles:

| Label              | Representative Skills                         |
|--------------------|-----------------------------------------------|
| `Data Analyst`     | SQL, Excel, Data Analysis, reporting          |
| `Data Scientist`   | Python, ML, statistics, Deep Learning         |
| `ML Engineer`      | Deep Learning, Java, model deployment         |
| `Project Manager`  | Leadership, Project Management, Communication |
| `Software Engineer`| Java, SQL, development                        |

The system implements a full ML pipeline:
data ingestion → feature engineering → TF-IDF vectorisation →
Logistic Regression → evaluation → Streamlit demo app.

---

## 2. Folder Structure

```
resume_screening/
├── data/
│   └── resume_dataset.csv         # Synthetic / curated dataset (300 rows)
├── models/
│   ├── resume_classifier.joblib   # Trained sklearn Pipeline  [auto-generated]
│   └── label_classes.json         # Label list                [auto-generated]
├── reports/
│   ├── metrics.json               # Evaluation metrics        [auto-generated]
│   └── confusion_matrix.png       # Confusion matrix plot     [auto-generated]
├── src/
│   ├── __init__.py
│   ├── config.py        — Central config (paths, hyperparameters)
│   ├── data_loader.py   — Data ingestion + feature engineering
│   ├── preprocessor.py  — Text normalisation (clean_text, stopwords)
│   ├── trainer.py       — Pipeline build, fit, save/load
│   ├── evaluator.py     — Metrics computation and plotting
│   ├── predictor.py     — Inference wrapper (ResumePredictor)
│   └── utils.py         — Shared helpers (JSON I/O, logging, dirs)
├── app/
│   └── streamlit_app.py — Streamlit web UI
├── notebooks/           — Exploratory analysis (optional)
├── train.py             — ← STEP 1: Run this to train
├── predict.py           — ← STEP 2 (optional): CLI inference
├── requirements.txt
└── README.md
```

---

## 3. Dataset & Limitations

- **File**: `data/resume_dataset.csv`
- **Rows**: 300 | **Columns**: Name, YearsExperience, Skills, Education, JobRole
- **Nature**: **Fully synthetic** — generated for educational NLP classification.

### Critical Dataset Limitation (Important for Interpreters)

This dataset shares **the same 10 skills across all 5 job roles** at near-equal
frequencies. Cross-validated macro F1 on even a Random Forest (the theoretical
ceiling) reaches only ~0.22 — barely above random chance (0.20 for 5 classes).

**This is a dataset property, not a model deficiency.** The pipeline is
architecturally correct and production-quality. The low metric accurately and
honestly reflects the lack of discriminating signal in the data.

**What this teaches:**
- Real-world ML always starts with data quality analysis
- Metrics must be reported honestly, even when unflattering
- A well-engineered pipeline on bad data still produces bad predictions
- Feature engineering cannot create signal that does not exist in the data

---

## 4. Installation

```bash
# 1. Enter the project directory
cd resume_screening

# 2. Create a virtual environment (strongly recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
```

No NLTK downloads required — stopwords are bundled inline.

---

## 5. How to Run (Step-by-Step)

### Step 1 — Train the Model

```bash
python train.py
```

**What happens:**
- Loads `data/resume_dataset.csv` and engineers features
- Stratified 80/20 train/validation split
- Trains ColumnTransformer pipeline (TF-IDF + numeric scaler) + Logistic Regression
- Prints full evaluation report (accuracy, macro F1, per-class breakdown)
- Saves model     → `models/resume_classifier.joblib`
- Saves metrics   → `reports/metrics.json`
- Saves CM plot   → `reports/confusion_matrix.png`

### Step 2 — CLI Prediction (Optional)

```bash
# Interactive prompt mode
python predict.py

# Pipe mode (returns JSON to stdout)
echo "Python, Deep Learning, Machine Learning, 4 years, Masters" | python predict.py
```

### Step 3 — Launch the Streamlit Web App

```bash
streamlit run app/streamlit_app.py
```

Open your browser at **http://localhost:8501**

Paste any resume text in the input box and click **Predict Job Role** to see:
- Predicted job category
- Confidence score
- Full probability bar chart across all classes

> **Note:** Run `python train.py` before launching the app.
> The app requires `models/resume_classifier.joblib` to exist.

---

## 6. Pipeline Architecture

```
CSV Input
  └─ data_loader.py
       ├─ TEXT_COLUMN  (skills text, lowercased, space-separated)
       ├─ years_exp    (numeric: years of experience)
       ├─ education_ord(ordinal: HighSchool=0 ... PhD=4)
       └─ skill_count  (numeric: number of skills listed)
            │
            ▼
     sklearn ColumnTransformer
       ├─ TEXT branch:
       │    TextCleaner (lowercase, remove stopwords/punctuation)
       │    └─ TfidfVectorizer (max_features=3000, ngram=(1,2), sublinear_tf)
       └─ NUMERIC branch:
            StandardScaler (years_exp, education_ord, skill_count)
            │
            ▼
     LogisticRegression
       (C=1.0, solver=lbfgs, class_weight=balanced, max_iter=2000)
            │
            ▼
     predict / predict_proba
```

All steps are wrapped in a single `sklearn.pipeline.Pipeline` for
reproducibility and to prevent data leakage during cross-validation.

---

## 7. Model Performance & Honest Reporting

| Metric          | Value  | Notes                                 |
|-----------------|--------|---------------------------------------|
| Accuracy        | ~0.18  | Near-random (5 classes, chance = 0.20)|
| Macro F1-Score  | ~0.18  | PRIMARY metric for multi-class        |
| Macro Precision | ~0.22  | Weighted by class frequency           |
| Macro Recall    | ~0.18  | Consistent with accuracy              |

**Why are metrics low?**
The dataset is fully synthetic with identical skill vocabularies across all
five roles. No feature — skills text, years of experience, or education level
— meaningfully separates the classes. This is verified empirically:
Random Forest cross-validation achieves the same ~0.22 ceiling.

**Responsible reporting principle:**
Inflating metrics via train-set evaluation, data leakage, or cherry-picked
thresholds is a common anti-pattern in AI demos. This project deliberately
reports honest validation-set metrics to model professional integrity.

Run `python train.py` to generate live metrics in `reports/metrics.json`.

---

## 8. Responsible AI — Ethical Limitations

### What This System Must NOT Be Used For

| Prohibited Use                              | Reason                                         |
|---------------------------------------------|------------------------------------------------|
| Real candidate screening or ranking         | Trained on synthetic data only                 |
| Automated rejection of job applications     | No fairness / bias audit performed             |
| Production hiring pipelines                 | Not validated on real-world resumes            |
| Comparing candidates across demographics    | Protected attributes not audited               |
| Confidence scores as candidate "scores"     | Probabilities are not calibrated for real use  |

### Known Bias Risks

1. **Skill vocabulary bias** — Certain skills may correlate with gender,
   ethnicity, or socioeconomic background in real-world data (e.g., "Leadership"
   skewing by cultural communication norms).

2. **Education privilege** — Ordinal encoding of education level treats
   higher degrees as inherently better, which can proxy socioeconomic advantage.

3. **Name / demographic leakage** — Candidate names are present in the raw CSV.
   Although excluded from this pipeline's features, any future model using names
   risks encoding gender or ethnicity bias.

4. **Synthetic distribution mismatch** — A model trained on synthetic uniform
   data will generalise poorly to real resumes with domain-specific vocabulary.

5. **No intersectional fairness analysis** — The model has NOT been audited
   for differential performance across demographic groups, as required by
   frameworks such as the EU AI Act for high-risk AI systems.

### Responsible Use Guidelines

- Use only as a **classroom exercise or portfolio demonstration**
- Always disclose to any audience that predictions are not validated for hiring
- If adapting for research, conduct a full fairness audit (disparate impact
  analysis, equalised odds) before any form of deployment
- Never present confidence scores as candidate quality rankings
- Log all predictions in extended experiments to detect systematic errors
- Consult legal counsel before deploying any ML system in a hiring context

### References

- [EEOC Guidance on AI in Hiring](https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines)
- [EU AI Act — High Risk AI](https://artificialintelligenceact.eu/the-act/)
- [AI Fairness 360 (IBM)](https://aif360.mybluemix.net/)
- Barocas, Hardt, Narayanan — *Fairness and Machine Learning* (2023)
- [NIST AI Risk Management Framework](https://www.nist.gov/system/files/documents/2023/01/26/AI%20RMF%201.0.pdf)

---

## 9. Tech Stack

| Component          | Library / Tool              | Version   |
|--------------------|-----------------------------|-----------|
| Data wrangling     | pandas, numpy               | 2.x, 1.x  |
| NLP preprocessing  | Built-in stopwords (no NLTK)| —         |
| Feature extraction | scikit-learn TF-IDF         | 1.3+      |
| Feature pipeline   | ColumnTransformer           | 1.3+      |
| Classifier         | Logistic Regression         | 1.3+      |
| Model persistence  | joblib                      | 1.3+      |
| Evaluation plots   | matplotlib, seaborn         | 3.7+      |
| Web UI             | Streamlit                   | 1.28+     |
| Python version     | 3.9+                        | —         |

---

*This project was created for educational purposes as Project 10 in an NLP / ML
portfolio. It demonstrates production pipeline design, honest metric reporting,
and responsible AI documentation alongside a working classification system.*
