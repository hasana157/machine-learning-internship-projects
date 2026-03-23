import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import ensure_dir

def run_preprocessing():
    # 1. Setup paths
    raw_path = "data/raw/insurance.csv"
    processed_dir = "data/processed"
    model_dir = "models"
    ensure_dir(processed_dir)
    ensure_dir(model_dir)

    # 2. Load raw data
    print("Loading raw data...")
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"Error: Could not find {raw_path}. Make sure the Kaggle file is in data/raw/")
        return

    # 3. Feature Engineering (Bands)
    print("Engineering age and BMI bands...")
    df['age_band'] = pd.cut(
        df['age'], 
        bins=[0, 30, 45, 60, 100], 
        labels=['18-30', '31-45', '46-60', '60+']
    )
    df['bmi_band'] = pd.cut(
        df['bmi'], 
        bins=[0, 25, 30, 35, 100], 
        labels=['normal', 'overweight', 'obese', 'severely_obese']
    )

    # 4. Define Features
    cat_features = ['sex', 'smoker', 'region', 'age_band', 'bmi_band']
    num_features = ['children', 'age', 'bmi']
    X = df.drop('charges', axis=1)
    y = df['charges']

    # 5. Build Preprocessor (ColumnTransformer)
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
        ('num', StandardScaler(), num_features)
    ])

    # 6. Split and Save
    print("Splitting and saving processed data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the split data
    X_train.to_csv(f"{processed_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{processed_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{processed_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{processed_dir}/y_test.csv", index=False)

    # Save the preprocessor object (required by train.py)
    joblib.dump(preprocessor, f"{model_dir}/preprocessor.joblib")
    print("Preprocessing complete! Files saved in data/processed/ and models/")

if __name__ == "__main__":
    run_preprocessing()