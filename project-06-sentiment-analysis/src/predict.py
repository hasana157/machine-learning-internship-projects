import argparse
import joblib
from src.utils import clean_text

def main(args):
    # Load the pipeline
    model = joblib.load(args.model_path)

    # Clean the input text
    cleaned = clean_text(args.text)

    # Predict
    pred = model.predict([cleaned])[0]
    sentiment = "Positive" if pred == 1 else "Negative"

    # Output probability if requested
    if args.probabilities:
        proba = model.predict_proba([cleaned])[0]
        print(f"Sentiment: {sentiment}")
        print(f"Negative probability: {proba[0]:.4f}")
        print(f"Positive probability: {proba[1]:.4f}")
    else:
        print(sentiment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment from text")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model (.joblib)")
    parser.add_argument("--text", type=str, required=True,
                        help="Input text to analyze")
    parser.add_argument("--probabilities", action="store_true",
                        help="Show prediction probabilities")
    args = parser.parse_args()
    main(args)