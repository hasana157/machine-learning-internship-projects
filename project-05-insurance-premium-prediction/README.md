# Insurance Premium Prediction & Error Analysis

## Project Overview
This project implements a regression pipeline to predict medical insurance premiums. Beyond simple prediction, it performs **segment-wise error analysis** to identify which customer profiles are most difficult for the model to price accurately.

## Key Technical Features
- **Feature Engineering**: Created custom bins for `Age` and `BMI` to capture non-linear risk factors.
- **Pipeline Architecture**: Utilized Scikit-Learn `ColumnTransformer` for automated preprocessing (OneHotEncoding + Scaling).
- **Model Comparison**: 
    - *Baseline*: Linear Regression
    - *Champion*: Gradient Boosting Regressor (Achieved significantly lower MAE).

## Technical Insights
- **The Multiplicative Effect**: EDA revealed that the interaction between `smoker` status and `BMI > 30` causes a non-linear spike in charges, which the Boosting model captured far better than the Linear baseline.
- **Error Analysis**: The model exhibits the highest MAE in the **Obese (BMI 35+)** and **Elderly (60+)** segments. This suggests that as risk factors compound, the variance in medical costs increases, making them "Hard-to-Predict" segments.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python -m src.train`
3. Run evaluation: `python -m src.evaluate`