import numpy as np
import pandas as pd
from pathlib import Path

p = Path('data/loan_applications.csv')
df = pd.read_csv(p)
rng = np.random.default_rng(42)

for col, fill in [
    ('employment_years', pd.Series(np.clip(rng.normal(7, 5, len(df)), 0, 40).round(1), index=df.index)),
    ('age', pd.Series(np.clip(rng.normal(38, 10, len(df)), 21, 70).round().astype(float), index=df.index)),
    ('previous_defaults', pd.Series(rng.choice([0, 1, 2], size=len(df), p=[0.78, 0.17, 0.05]), index=df.index)),
    ('existing_loans_count', pd.Series(rng.choice([0, 1, 2, 3], size=len(df), p=[0.5, 0.3, 0.15, 0.05]), index=df.index)),
    ('credit_score', pd.Series(np.clip(rng.normal(650, 80, len(df)), 300, 850).round(), index=df.index)),
]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(fill)

for col in ['education', 'self_employed', 'property_area', 'marital_status']:
    if col in df.columns:
        df[col] = df[col].astype('object')
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

if 'loan_approved' in df.columns:
    df['loan_approved'] = pd.to_numeric(df['loan_approved'], errors='coerce').astype(float)

# Keep only rows with a target value
if 'loan_approved' in df.columns:
    df = df.dropna(subset=['loan_approved'])

df.to_csv(p, index=False)
print('saved', p)
print(df.isnull().mean().sort_values(ascending=False).head())
print('rows', len(df))
