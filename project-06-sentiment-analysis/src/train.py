import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.utils import clean_text, ensure_dir, save_json

def load_and_clean_data(data_path):
    """Load raw CSV, clean reviews, encode labels."""
    df = pd.read_csv(data_path)
    # Clean reviews (apply the same function as in utils)
    df['cleaned_review'] = df['review'].apply(lambda x: clean_text(x))
    # Encode labels: positive -> 1, negative -> 0
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    # Drop any rows where cleaning produced empty text
    df = df[df['cleaned_review'] != ''].reset_index(drop=True)
    return df['cleaned_review'], df['label']

def main(args):
    print("Loading and cleaning data...")
    X, y = load_and_clean_data(args.data_path)
    print(f"Data loaded: {len(X)} samples")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=args.max_features,
            ngram_range=(args.ngram_min, args.ngram_max)
        )),
        ('clf', LogisticRegression(
            C=args.C,
            penalty=args.penalty,
            max_iter=1000,
            random_state=args.random_state,
            solver='liblinear' if args.penalty == 'l1' else 'lbfgs'
        ))
    ])

    # Optional grid search
    if args.grid_search:
        print("Running grid search (this may take a while)...")
        param_grid = {
            'tfidf__max_features': [3000, 5000, 10000],
            'tfidf__ngram_range': [(1,1), (1,2)],
            'clf__C': [0.1, 1, 10]
        }
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)
        best_cv_f1 = grid_search.best_score_
    else:
        # Train with specified parameters
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_cv_f1 = None

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest F1 Score: {f1:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    ensure_dir(args.out_dir)
    model_path = f"{args.out_dir}/sentiment_pipeline.joblib"
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

    # Save metrics
    metrics = {
        'test_f1': f1,
        'test_accuracy': accuracy,
        'params': {
            'max_features': args.max_features,
            'ngram_range': f"({args.ngram_min},{args.ngram_max})",
            'C': args.C,
            'penalty': args.penalty
        }
    }
    if args.grid_search:
        metrics['best_cv_f1'] = best_cv_f1
        metrics['best_params'] = grid_search.best_params_

    metrics_path = f"{args.out_dir}/metrics.json"
    save_json(metrics_path, metrics)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument("--data_path", type=str, default="data/raw/IMDB Dataset.csv",
                        help="Path to raw CSV dataset")
    parser.add_argument("--out_dir", type=str, default="models",
                        help="Directory to save model and metrics")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data for testing")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max_features", type=int, default=5000,
                        help="Maximum number of TF‑IDF features")
    parser.add_argument("--ngram_min", type=int, default=1,
                        help="Minimum n-gram size")
    parser.add_argument("--ngram_max", type=int, default=1,
                        help="Maximum n-gram size")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse regularization strength")
    parser.add_argument("--penalty", type=str, default="l2",
                        choices=["l1", "l2"], help="Regularization penalty")
    parser.add_argument("--grid_search", action="store_true",
                        help="Perform grid search (overrides manual params)")

    args = parser.parse_args()
    main(args)