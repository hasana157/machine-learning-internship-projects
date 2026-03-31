# src/predict.py
import joblib
import re

def clean_text(text):
    """Clean and normalize text (same as training)."""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def predict_spam(model_path, text):
    """
    Predict if a single text is spam or not and show top contributing words.
    """
    model = joblib.load(model_path)
    text_clean = clean_text(text)
    pred = model.predict([text_clean])[0]
    label = "Spam" if pred == 1 else "Not Spam"

    # Top contributing words
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]

    words = text_clean.split()
    contrib = {word: coefs[tfidf.vocabulary_[word]] for word in words if word in tfidf.vocabulary_}
    top_words = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return label, top_words

# CLI-like usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    prediction, top_words = predict_spam(args.model_path, args.text)
    print("Prediction:", prediction)
    print("Top contributing words:", top_words)