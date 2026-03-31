# src/train.py
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from src.utils import ensure_dir, save_json, load_data
import re

# -------------------------
# Text preprocessing
# -------------------------
def clean_text(text):
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text

# -------------------------
# Training pipeline
# -------------------------
def train_model(data_path, out_dir="models"):
    # Load and preprocess data
    df = load_data(data_path)
    df['clean_text'] = df['text'].apply(clean_text)

    X = df['clean_text']
    y = df['label']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Hyperparameter tuning
    param_grid = {
        "tfidf__max_features": [1000, 2000, 3000],
        "clf__C": [0.1, 1, 10]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("Best Parameters:", grid.best_params_)

    # Evaluate
    preds = best_model.predict(X_test)
    f1 = f1_score(y_test, preds)
    print("F1 Score on Test Set:", f1)

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, target_names=['Ham', 'Spam']))

    # Save model & metrics
    ensure_dir(out_dir)
    joblib.dump(best_model, f"{out_dir}/spam_pipeline.joblib")
    save_json(f"{out_dir}/metrics.json", {"f1_score": float(f1), "best_params": grid.best_params_})

    # -------------------------
    # Feature importance / top spam words
    # -------------------------
    tfidf = best_model.named_steps["tfidf"]
    clf = best_model.named_steps["clf"]

    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]
    top_spam_idx = np.argsort(coefs)[-20:]  # top 20 words indicative of spam
    top_spam_words = feature_names[top_spam_idx]
    top_spam_scores = coefs[top_spam_idx]

    ensure_dir("reports/figures")
    plt.figure(figsize=(10,5))
    plt.barh(top_spam_words, top_spam_scores, color='orange')
    plt.title("Top 20 Words Indicative of Spam")
    plt.tight_layout()
    plt.savefig("reports/figures/top_spam_words.png", dpi=150)
    plt.close()

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=150)
    plt.close()

    print("Model and metrics saved successfully!")

# -------------------------
# CLI entry point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/raw/spam.tsv")
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    train_model(args.data_path, args.out_dir)