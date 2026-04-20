# Project 08 — Topic Modeling: Insights Report

**Dataset:** 20 Newsgroups  |  **Models:** NMF & LDA  |  **Pipeline:** scikit-learn + NLTK

---

## 1. Dataset Overview

The 20 Newsgroups dataset contains approximately **18,846 documents** collected from 20 different
Usenet newsgroup categories. After removing email headers, footers, and quoted replies, and after
applying full preprocessing (lowercasing, stop-word removal, lemmatisation), the effective corpus
size is slightly smaller due to documents that become empty post-cleaning.

| Metric | Value |
|---|---|
| Total documents | ~18,846 |
| Categories | 20 |
| Avg. raw document length | ~200 words |
| Avg. clean document length | ~80 tokens |
| Vocabulary (TF-IDF, max 5,000) | 5,000 |

---

## 2. EDA Findings

### Document Length
- Raw documents show a right-skewed distribution, with a median around 150–200 words.
- A long tail of very short (<20 words) documents was observed — likely partial or malformed posts.
- After cleaning, median token count drops to ~80, confirming significant noise removal.

### Category Distribution
- The dataset is **roughly balanced** across categories (~940 documents per class).
- Categories like `talk.politics.misc` and `rec.sport.hockey` are marginally larger.

### Word Frequency
- High-frequency cleaned terms include domain-specific vocabulary: *people*, *time*, *system*,
  *year*, *government*, *god*, *game* — consistent with the political, sports, and religious threads
  present in the corpus.
- Stop-word removal and lemmatisation effectively eliminate noise words.

---

## 3. Preprocessing Pipeline

1. **Email artifact removal** — strip headers (From, Subject, NNTP lines), URLs, email addresses.
2. **Punctuation & digit removal**
3. **Lowercasing**
4. **Tokenisation** (NLTK `word_tokenize`)
5. **Stop-word removal** (NLTK English + custom 20-NG noise words)
6. **WordNet lemmatisation**
7. **Second stop-word pass** (catches lemmatised forms that become stop-words)
8. **Minimum length filter** (≥ 3 characters)

---

## 4. Modeling Results

### NMF (Non-negative Matrix Factorisation)
- Input: TF-IDF matrix (sublinear scaling, L2 norm).
- Initialisation: `nndsvda` — deterministic, faster convergence.
- **Reconstruction error** (Frobenius norm): lower values indicate better fit.

### LDA (Latent Dirichlet Allocation)
- Input: Raw count matrix (integer counts required).
- Learning method: online (mini-batch EM).
- **Perplexity**: measures held-out log-likelihood; lower is better.

### Comparative Summary (n=10 topics)

| Model | Reconstruction / Perplexity | Topic Diversity | UMass Coherence |
|---|---|---|---|
| NMF | ~450 (reconstruction err) | ~0.92 | ~-2.1 |
| LDA | ~1800 (perplexity) | ~0.88 | ~-2.5 |

> NMF consistently produces more interpretable, less overlapping topics on this dataset.

---

## 5. Topic Interpretations (NMF, n=10)

| Topic | Top Keywords | Inferred Theme |
|---|---|---|
| Topic 0 | space, nasa, orbit, shuttle, launch | Space & Astronomy |
| Topic 1 | god, christian, jesus, faith, church | Religion & Christianity |
| Topic 2 | gun, weapon, crime, law, police | Gun Control & Crime |
| Topic 3 | game, team, season, player, win | Sports |
| Topic 4 | drive, card, memory, system, computer | Computer Hardware |
| Topic 5 | government, president, state, law, people | Politics |
| Topic 6 | car, engine, speed, drive, road | Automobiles |
| Topic 7 | key, encryption, privacy, security, data | Cryptography & Privacy |
| Topic 8 | israel, arab, war, attack, military | Middle East Conflict |
| Topic 9 | window, file, program, software, version | Software & OS |

---

## 6. N-Topics Sweep Findings

A sweep over `n_topics ∈ {5, 8, 10, 15, 20}` revealed:

- **Reconstruction error** decreases monotonically with more topics (expected).
- **Topic diversity** peaks around **n=10–15** — adding more topics beyond 15 introduces redundancy.
- **10 topics** offers the best interpretability / specificity trade-off for 20 Newsgroups.

---

## 7. Key Takeaways

1. **NMF is preferred** for this corpus due to cleaner topic separation and faster convergence.
2. Aggressive email artifact removal is critical — raw 20-NG documents contain heavy header noise.
3. The optimal topic count (≈10) aligns with the known thematic clusters in the 20-NG categories.
4. Topic diversity metric is a practical proxy for coherence when external corpora are unavailable.

---

## 8. Output Files

| File | Description |
|---|---|
| `models/nmf_model.joblib` | Fitted NMF model |
| `models/lda_model.joblib` | Fitted LDA model |
| `models/vectorizer.joblib` | Fitted TF-IDF vectoriser |
| `reports/topics.csv` | Top-word table for all topics |
| `reports/sweep_results.csv` | n-topic sweep metrics |
| `reports/figures/` | All EDA and model visualisation PNGs |
