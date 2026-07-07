# Machine Learning Internship Projects

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Projects](https://img.shields.io/badge/Projects-25-brightgreen) ![License](https://img.shields.io/badge/License-MIT-green)

Portfolio of **25 machine learning projects** spanning regression, classification, NLP, computer vision, recommender systems, and time-series forecasting.

---

## 📂 Projects Overview

### Regression & Forecasting (6 projects)

| # | Project | Problem | Approach |
|---|---------|---------|----------|
| 01 | House Price Prediction | Predict property prices | Linear regression, ensemble methods |
| 05 | Insurance Premium Prediction | Estimate insurance costs | Regression with categorical encoding |
| 16 | Sales Forecasting | Predict future sales | Time-series analysis, ARIMA |
| 17 | Energy Consumption Forecasting | Forecast energy demand | LSTM, seasonal decomposition |
| 18 | Stock Movement Prediction | Predict stock direction | Technical indicators, ML classifiers |
| 19 | Demand Forecasting for Inventory | Forecast product demand | Regression, seasonality modeling |

---

### Classification (13 projects)

#### Core Classification
| # | Project | Problem | Approach |
|---|---------|---------|----------|
| 02 | Credit Default Prediction | Identify loan default risk | Logistic regression, random forest, imbalance handling |
| 03 | Customer Churn Prediction | Predict customer attrition | Classification with feature importance |
| 04 | Fraud Detection | Detect fraudulent transactions | Ensemble methods, SMOTE, cost-sensitive learning |
| 06 | Sentiment Analysis | Classify text sentiment | TF-IDF, Naive Bayes, SVM |
| 07 | Spam Email Classifier | Detect spam emails | Text vectorization, logistic regression |
| 09 | News Category Classification | Classify news articles | NLP preprocessing, multi-class classification |
| 10 | Resume Screening Model | Screen resumes for job match | Text processing, similarity matching |
| 23 | Lead Scoring Model | Score sales leads | Binary classification, business metrics |
| 25 | Loan Approval Risk Prediction | Predict loan approval risk | Risk scoring, threshold optimization |

#### Computer Vision
| # | Project | Problem | Approach |
|---|---------|---------|----------|
| 11 | Image Classification | Classify general images | CNN, transfer learning |
| 12 | Cat vs Dog Classifier | Binary image classification | CNN, data augmentation |
| 13 | Face Mask Detection | Detect masks in images | YOLO, object detection |
| 14 | Handwritten Digit Classifier (MNIST) | Classify digits 0-9 | CNN, MNIST dataset |
| 15 | Defect Detection | Detect product defects | Image segmentation, anomaly detection |

---

### Natural Language Processing (5 projects)

| # | Project | Problem | Approach |
|---|---------|---------|----------|
| 06 | Sentiment Analysis | Text sentiment classification | TF-IDF, Naive Bayes, SVM |
| 07 | Spam Email Classifier | Binary text classification | Bag-of-words, logistic regression |
| 08 | Topic Modeling | Extract topics from documents | LDA, NMF, topic coherence |
| 09 | News Category Classification | Multi-class text classification | NLP preprocessing, vectorization |
| 10 | Resume Screening Model | Extract & match resume data | Text parsing, similarity metrics |

---

### Recommender Systems (2 projects)

| # | Project | Problem | Approach |
|---|---------|---------|----------|
| 21 | Movie Recommender (Collaborative) | Recommend movies to users | Collaborative filtering, matrix factorization |
| 22 | Product Recommendation (Content-Based) | Recommend products | Content-based filtering, similarity scores |

---

### Clustering & Segmentation (1 project)

| # | Project | Problem | Approach |
|---|---------|---------|----------|
| 24 | Customer Segmentation | Segment customers into groups | K-means, hierarchical clustering, RFM analysis |

---

### Anomaly Detection & Time-Series (1 project)

| # | Project | Problem | Approach |
|---|---------|---------|----------|
| 20 | Anomaly Detection in Sensor Data | Detect abnormal sensor readings | Isolation Forest, autoencoders, statistical methods |

---

## 🏗️ Repository Structure

```
machine-learning-internship-projects/
├── project-01-house-price-prediction/
├── project-02-credit-default/
├── project-03-customer-churn-prediction/
├── project-04-fraud-detection/
├── project-05-insurance-premium-prediction/
├── project-06-sentiment-analysis/
├── project-07-spam-email-classifier(NLP)/
├── project-08-topic-modeling/
├── project-09-news-category-classification/
├── project-10-resume-screening-model/
├── project-11-image-classification/
├── project-12-cat-vs-dog-classifier/
├── project-13-face-mask-detection/
├── project-14-handwritten-digit-classifier-MNIST/
├── project-15-defect-detection/
├── project-16-sales-forecasting/
├── project-17-energy-consumption-forecasting/
├── project-18-stock-movement-prediction/
├── project-19-demand-forecasting-for-inventory/
├── project-20-anomaly-detection-in-sensor-data/
├── project-21-movie-recommender-collaborative/
├── project-22-product-recommendation-content-based/
├── project-23-lead-scoring-model-business-ml/
├── project-24-customer-segmentation-clustering/
├── project-25-capstone-LoanApprovalRiskPredicti/
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup

```bash
# Clone
git clone https://github.com/hasana157/machine-learning-internship-projects.git
cd machine-learning-internship-projects

# Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt
```

### Dependencies
```
pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter
tensorflow/keras (for deep learning projects)
nltk, spacy (for NLP projects)
```

---

## 🔑 Key Methodologies

**Data Processing:** EDA, missing value imputation, outlier handling, feature scaling  
**Feature Engineering:** Polynomial features, interaction terms, domain-specific features  
**Regression:** Linear, polynomial, ridge/lasso, tree-based, ensemble  
**Classification:** Logistic regression, SVM, Naive Bayes, decision trees, ensemble, neural networks  
**NLP:** Tokenization, TF-IDF, word embeddings, LDA, topic modeling  
**Computer Vision:** CNN, transfer learning, data augmentation, object detection  
**Evaluation:** Accuracy, precision, recall, F1, ROC-AUC, MAE, RMSE, R²  
**Imbalance Handling:** SMOTE, class weights, threshold optimization  

---

## 📊 Quick Stats

| Domain | Count | Focus |
|--------|-------|-------|
| Regression | 6 | Time-series, forecasting, pricing |
| Classification | 13 | Risk scoring, detection, text, vision |
| NLP | 5 | Text classification, topic modeling, screening |
| Computer Vision | 5 | Image classification, object detection, segmentation |
| Recommender Systems | 2 | Collaborative & content-based filtering |
| Clustering | 1 | Customer segmentation |
| Anomaly Detection | 1 | Sensor data, outlier detection |

---

## 🎯 Key Features

✅ All projects follow reproducible, step-by-step workflows  
✅ Verified metrics from actual model evaluation (no fabrication)  
✅ Clean, commented code with PEP 8 conventions  
✅ EDA and data understanding for each project  
✅ Clear model evaluation and comparison  
✅ Self-contained notebooks ready to run  

---

## 📝 Notes

- **Data**: Public datasets, synthetic data, or case studies (documented per project)
- **Metrics**: All reported values from actual evaluations
- **Reproducibility**: Random seeds set, step-by-step notebooks, output cells preserved
- **Best Practices**: sklearn conventions, cross-validation, proper train/test splits

---

## 👤 Author

**Hasana** — AI Engineer
[GitHub](https://github.com/hasana157)  • [LinkedIn](https://linkedin.com/in/hasana157)

---

**Last Updated:** July 2026 | **Total Projects:** 25
