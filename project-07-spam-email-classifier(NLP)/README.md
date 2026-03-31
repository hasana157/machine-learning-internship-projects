
# 📩 Spam Email Classifier

## **1. Project Overview**

This project demonstrates a **Machine Learning-based SMS and Email spam detection system**. The goal is to automatically classify messages as **Spam** or **Not Spam (Ham)** using **TF-IDF feature extraction** and **Logistic Regression**.

The project highlights skills in **NLP, ML pipelines, Streamlit web apps, data-driven insights, and production-ready deployment**.

**Pipeline Includes:**

* Data Exploration & Cleaning
* Feature Engineering (TF-IDF + Bigrams + Message Length)
* Model Training & Evaluation
* Advanced Web App for Real-time Detection
* Batch CSV Prediction
* Interpretability via Top Contributing Words

---

## **2. Project Structure**

A professional, organized structure for clarity and scalability:

```text
spam_email_classifier/
│
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── raw/                   # Original SMS dataset (e.g., spam.csv)
│   └── processed/             # Preprocessed CSV for modeling
│
├── models/
│   └── spam_pipeline.joblib   # Saved Logistic Regression pipeline
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_preprocessing.ipynb # Text cleaning & feature engineering
│   ├── 03_evaluate.ipynb      # Model evaluation & metrics
│   └── 04_predict.ipynb       # Prediction & top word analysis
│
├── reports/
│   ├── length_distribution.png
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── app.jpg                # Screenshot of the Streamlit app
│   └── batch_ui.png           # Screenshot of batch upload interface
│
├── src/
│   ├── app.py                 # Original Streamlit app
│   ├── advanced_app_v2.py     # Professional dark-themed Streamlit app
│   ├── preprocess.py          # Text cleaning / feature functions
│   ├── predict.py             # Prediction logic / top word contribution
│   └── train_model.py         # Model training + hyperparameter tuning
```

**Notes:**

* `data/` → Keep raw and processed datasets separate.
* `models/` → Store serialized models for deployment.
* `notebooks/` → Step-by-step experiments, EDA, preprocessing, evaluation, and prediction.
* `reports/` → Figures, charts, and app screenshots.
* `src/` → All Python scripts for app, preprocessing, and training.

---

## **3. Dataset Overview**

The dataset contains **5,572 SMS messages**, labeled as **Ham (legitimate)** or **Spam**.

| Label    | Count | Description                  |
| -------- | ----- | ---------------------------- |
| Ham (0)  | 4,825 | Legitimate messages          |
| Spam (1) | 747   | Junk or unsolicited messages |

### **3.1 Message Length Analysis**

* Ham messages: Mostly short (0–50 characters).
* Spam messages: Cluster around 130–160 characters.

![Message Length Distribution](reports/length_distribution.png)

### **3.2 Word Analysis**

* **Spam Vocabulary:** `free`, `claim`, `prize`, `cash`, `won`
* **Ham Vocabulary:** `u`, `ok`, `love`, `time`, `day`

![Class Distribution](reports/class_distribution.png)

**Insight:** Message length and word choice are strong differentiators for Spam vs. Ham.

---

## **4. Preprocessing & Feature Engineering**

* **Text Cleaning:** Lowercasing, removing punctuation & extra spaces.
* **Features:**

  * **Message Length** – numeric feature.
  * **TF-IDF Vectorization** – converts text into weighted features.
  * **Bigrams (`ngram_range=(1,2)`)** – captures phrases like `"cash prize"`.

---

## **5. Modeling**

* **Model:** Logistic Regression with TF-IDF input.
* **Hyperparameter Tuning:** Grid search on `C` and `max_features`.
* **Best Parameters:** `{'clf__C': 10, 'tfidf__max_features': 2000}`
* **F1 Score on Test Set:** 0.912

The model balances detection of Spam while minimizing false positives on Ham.

---

## **6. Evaluation**

### **6.1 Confusion Matrix**

![Confusion Matrix](reports/confusion_matrix.png)

* **True Negatives:** 954
* **True Positives:** 135
* **False Positives:** 12
* **False Negatives:** 14

### **6.2 Metrics Summary**

| Class                | Precision | Recall | F1-score  | Support |
| -------------------- | --------- | ------ | --------- | ------- |
| Ham (0)              | 0.986     | 0.988  | 0.987     | 966     |
| Spam (1)             | 0.918     | 0.906  | 0.912     | 149     |
| **Overall Accuracy** |           |        | **0.977** | 1115    |

---

## **7. Advanced Web App (Streamlit)**

### **7.1 Features**

* Real-time detection with **dark “detective-style” UI**.
* **Batch CSV upload** for multiple messages.
* **Prediction confidence** with progress bars.
* **Top contributing words** for interpretability.
* **Predict / Reset buttons** for smooth UX flow.

### **7.2 Screenshots**

**Main UI:**
![App Screenshot](reports/app.jpg)

**Batch Upload:**
![Batch Upload Placeholder](reports/batch_ui.png)

---

## **8. Sample Predictions & Interpretability**

| Text                                           | Prediction | Top Words                                                         |
| ---------------------------------------------- | ---------- | ----------------------------------------------------------------- |
| "Congratulations! Claim your free voucher now" | Spam 🚨    | claim (2.70), voucher (1.45), free (1.21), congratulations (0.53) |
| "Hey, can you call me later?"                  | Not Spam ✅ | later (-2.23), hey (-2.02)                                        |
| "You have WON 150p, text STOP to unsubscribe"  | Spam 🚨    | 150p (3.38), stop (1.92), won (1.91)                              |
| "Let's meet for coffee at 5"                   | Not Spam ✅ | let (-0.78), coffee (-0.45), meet (0.01)                          |

---

## **9. Conclusion**

* **ML Pipeline:** Clean → TF-IDF + Bigrams → Logistic Regression.
* **Performance:** 97.6% accuracy, balanced precision/recall.
* **Deployment Ready:** Streamlit app with batch predictions and visual interpretability.
* **Resume Impact:** Shows **NLP, ML modeling, data visualization, and full-stack deployment skills**.

---

## **10. Future Improvements**

* Ensemble models for more **robust predictions**.
* SHAP/LIME integration for **deep interpretability**.
* Convert app into **Flask API + front-end** for production deployment.
* Integrate **real-time SMS/email API** for automated filtering.


