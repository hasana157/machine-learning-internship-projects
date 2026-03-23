import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from src.utils import ensure_dir

def train_model():
    # Define paths
    processed_dir = "data/processed"
    model_dir = "models"
    ensure_dir(model_dir)

    print("Loading processed data...")
    X_train = pd.read_csv(f"{processed_dir}/X_train.csv")
    y_train = pd.read_csv(f"{processed_dir}/y_train.csv").values.ravel()
    
    # Load the preprocessor created during the experimentation phase
    try:
        preprocessor = joblib.load(f"{model_dir}/preprocessor.joblib")
    except FileNotFoundError:
        print("Error: preprocessor.joblib not found in models/. Run the preprocessing notebook first.")
        return

    # Initialize the Gradient Boosting Regressor
    # We use these params based on our EDA findings regarding non-linear interactions
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    # Create the full pipeline
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', gb_model)
    ])

    print("Training the Gradient Boosting model...")
    pipeline.fit(X_train, y_train)

    # Save the final model
    model_path = f"{model_dir}/gb_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_model()