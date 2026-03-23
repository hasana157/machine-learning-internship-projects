import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from src.utils import save_json, ensure_dir

def run_evaluation():
    # Define paths
    model_path = "models/gb_model.joblib"
    test_data_path = "data/processed/X_test.csv"
    test_labels_path = "data/processed/y_test.csv"
    reports_dir = "reports"
    ensure_dir(reports_dir)

    print("Loading model and test data...")
    model = joblib.load(model_path)
    X_test = pd.read_csv(test_data_path)
    y_test = pd.read_csv(test_labels_path).values.ravel()

    # Generate predictions
    preds = model.predict(X_test)
    
    # Calculate global metrics
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    metrics = {
        "overall_mae": round(mae, 2),
        "r2_score": round(r2, 4)
    }
    save_json(f"{reports_dir}/metrics.json", metrics)

    # Segment-wise Error Analysis
    eval_df = X_test.copy()
    eval_df['actual'] = y_test
    eval_df['predicted'] = preds
    eval_df['abs_error'] = (eval_df['actual'] - eval_df['predicted']).abs()

    # Calculate MAE per Age and BMI band
    segment_mae = (
        eval_df.groupby(['age_band', 'bmi_band'], observed=True)
        .agg(mae=('abs_error', 'mean'), count=('abs_error', 'count'))
        .reset_index()
        .sort_values(by='mae', ascending=False)
    )

    # Save segments to CSV
    segment_mae.to_csv(f"{reports_dir}/segment_mae.csv", index=False)
    
    print("\nEvaluation Complete.")
    print(f"Overall MAE: {metrics['overall_mae']}")
    print("Top 5 highest error segments:")
    print(segment_mae.head())

if __name__ == "__main__":
    run_evaluation()