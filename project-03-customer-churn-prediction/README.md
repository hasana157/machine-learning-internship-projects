# Project 03 — Customer Churn Prediction

## Overview

Customer churn is one of the most important problems for subscription-based businesses. Losing customers directly impacts revenue, so companies try to identify customers who are likely to leave and intervene early.

This project builds a **machine learning system to predict customer churn** using tabular business data.
The pipeline includes data preprocessing, model training, model evaluation, and explainability using permutation importance.

The final output highlights the **top drivers behind customer churn**, which can help businesses design retention strategies.

---

## Project Objectives

* Build a churn prediction model using tabular customer data
* Compare a baseline model with a tree-based model
* Identify the most important factors contributing to churn
* Produce interpretable insights for business decision-making

---

## Dataset

A synthetic dataset was generated to simulate real telecom customer data.

Features include:

| Feature          | Type        | Description                                     |
| ---------------- | ----------- | ----------------------------------------------- |
| tenure_months    | Numerical   | How long the customer has been with the company |
| monthly_charges  | Numerical   | Monthly service cost                            |
| contract_type    | Categorical | Contract duration type                          |
| internet_service | Categorical | Type of internet service                        |
| payment_method   | Categorical | Customer payment method                         |
| churn            | Target      | 1 = customer left, 0 = customer stayed          |

The dataset contains **5000 samples**.

---

## Project Structure

```
project-03-customer-churn-prediction

data/
  raw/
    churn_data.csv

notebooks/
  01_data_generation.ipynb
  02_eda.ipynb
  03_model_experiments.ipynb

src/
  utils.py
  train.py
  evaluate.py

models/
  logreg_model.joblib
  rf_model.joblib
  test_split.joblib

reports/
  figures/
    churn_distribution.png
    contract_vs_churn.png
    charges_vs_churn.png
    churn_drivers.png
  metrics.json
  top_churn_drivers.csv
  eda_summary.json
```

---

## Machine Learning Pipeline

The project follows a standard machine learning workflow:

1. Data loading
2. Feature preprocessing
3. Model training
4. Model evaluation
5. Model explainability
6. Report generation

### Preprocessing

A preprocessing pipeline was built using **ColumnTransformer**:

* Numerical features → StandardScaler
* Categorical features → OneHotEncoder

This ensures that both feature types are correctly handled before training.

---

## Models

Two models were trained and compared:

### Baseline Model

Logistic Regression

This provides a simple interpretable baseline for churn prediction.

### Improved Model

Random Forest Classifier

Tree-based models typically perform better on tabular business datasets.

---

## Evaluation Metric

The main evaluation metric used is:

**F1 Score**

F1 score balances both precision and recall, making it suitable for churn prediction where class imbalance may exist.

Example result (Logistic Regression):

```
F1 Score ≈ 0.41
```

---

## Model Explainability

To understand what drives churn predictions, **Permutation Importance** was used.

Permutation importance works by:

1. Measuring model performance
2. Randomly shuffling one feature
3. Measuring performance drop
4. Larger drop = more important feature

This method is **model-agnostic**, meaning it works with any machine learning model.

---

## Key Churn Drivers

Top factors contributing to customer churn:

1. Month-to-month contracts
2. High monthly charges
3. Low customer tenure
4. Internet service type
5. Payment method

These results suggest that **new customers with expensive monthly plans and flexible contracts are more likely to churn**.

---

## Business Insights

From the model results we can derive several practical insights:

* Customers on **month-to-month contracts** show the highest churn risk.
* Customers with **high monthly charges** are more likely to leave.
* **New customers** (low tenure) churn more frequently.

Potential retention strategies:

* Offer discounts for long-term contracts
* Provide loyalty incentives for new customers
* Monitor high-charge customers for churn risk

---

## How to Run the Project

### Install dependencies

```
pip install -r requirements.txt
```

---

### Train baseline model

```
python -m src.train --model_type logreg
```

---

### Train Random Forest model

```
python -m src.train --model_type rf
```

---

### Evaluate model and generate churn drivers report

```
python -m src.evaluate --model_path models/rf_model.joblib
```

---

## Generated Artifacts

After running the pipeline the following outputs are created:

* Model files in `models/`
* Performance metrics in `reports/metrics.json`
* Feature importance report in `reports/top_churn_drivers.csv`
* Visualization of churn drivers in `reports/figures/churn_drivers.png`

These outputs provide both **technical evaluation and business insight**.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

---

## Future Improvements

Potential improvements for this project include:

* Hyperparameter tuning
* SHAP feature explanations
* ROC and Precision-Recall curves
* Deployment as an API
* Dashboard for churn monitoring

---

## Conclusion

This project demonstrates how machine learning can be used to predict customer churn and identify the key factors influencing customer retention. By combining predictive modeling with explainability techniques, the system provides actionable insights that businesses can use to reduce churn and improve customer loyalty.
