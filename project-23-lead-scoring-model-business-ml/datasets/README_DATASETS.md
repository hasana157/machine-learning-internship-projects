# LeadForge AI - Dataset Guide

## 📊 Primary Dataset: IBM HR Employee Attrition

### Why This Dataset?

The IBM HR Employee Attrition dataset is the **recommended** choice for LeadForge AI because:

✅ **Perfect binary classification proxy** - Employee attrition mirrors lead conversion  
✅ **1,470 well-documented records** - Real-world data quality  
✅ **35+ features** - Rich feature engineering opportunities  
✅ **No missing values** - Ready to use out of the box  
✅ **Class imbalance** (~16% attrition) - Realistic sales scenario  
✅ **Widely trusted** - Used in academic and industry ML projects  

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Records** | 1,470 employees |
| **Target** | Attrition (Yes/No) |
| **Features** | 35 total (16 numeric, 19 categorical) |
| **Positive Class** | 237 (16.1%) - good imbalance ratio |
| **Negative Class** | 1,233 (83.9%) |
| **Missing Values** | 0 |
| **Data Types** | Integer, Float, String |
| **File Size** | ~1.5 MB |

---

## 🔗 Download Instructions

### Option 1: Direct Kaggle Download (Recommended)

**Step-by-Step:**

1. **Create Kaggle Account**
   - Go to https://www.kaggle.com/signup
   - Register with email or Google account

2. **Download Dataset**
   - Navigate to: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
   - Click **"Download"** button (top-right)
   - Save as `ibm_attrition.zip`

3. **Extract Files**
   ```bash
   unzip ibm_attrition.zip
   cd datasets/raw/
   # You should see: WA_Fn-UseC_-HR-Employee-Attrition.csv
   ```

4. **Verify Download**
   ```bash
   ls -lh WA_Fn-UseC_-HR-Employee-Attrition.csv
   # Should be ~1.5 MB
   ```

### Option 2: Kaggle API (Advanced)

**Install Kaggle CLI:**
```bash
pip install kaggle

# Setup API credentials:
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/
# 4. chmod 600 ~/.kaggle/kaggle.json
```

**Download via CLI:**
```bash
kaggle datasets download -d pavansubhasht/ibm-hr-analytics-attrition-dataset
unzip ibm-hr-analytics-attrition-dataset.zip
```

### Option 3: Generate Synthetic B2B CRM Data

If you can't download from Kaggle, generate synthetic data:

```bash
cd datasets
python generate_synthetic_crm.py --n_leads 50000 --output train_data.csv
```

---

## 📋 Dataset Features

### Demographic Features
```
• Age (numeric): 18-65 years
• Gender (categorical): Male, Female
• MaritalStatus (categorical): Single, Married, Divorced
• Department (categorical): Sales, Research & Development, Human Resources
```

### Job-Related Features
```
• JobRole (categorical): Sales Executive, Manager, Analyst, etc.
• JobLevel (numeric): 1-5 (1=entry, 5=director)
• JobSatisfaction (numeric): 1-4 (satisfaction rating)
• YearsAtCompany (numeric): 0-40 years
• YearsInRole (numeric): 0-18 years
• YearsInCurrentRole (numeric): tenure in current position
```

### Performance & Compensation
```
• PerformanceRating (numeric): 3-4 (1-4 scale)
• MonthlyIncome (numeric): $1000-$20000
• PercentSalaryHike (numeric): 11-25%
• StockOptionLevel (numeric): 0-3
• TrainingTimesLastYear (numeric): 0-6 trainings
```

### Engagement & Health
```
• EnvironmentSatisfaction (numeric): 1-4
• WorkLifeBalance (numeric): 1-4
• OverTime (categorical): Yes, No
• DailyRate, HourlyRate, MonthlyRate (numeric): Compensation details
• DistanceFromHome (numeric): 1-29 miles
```

### Target Variable
```
• Attrition (binary): Yes (1) / No (0)
  - Maps to Lead Conversion in B2B context
  - "Attrition=Yes" → "High engagement/likely to convert"
  - "Attrition=No" → "Low engagement/unlikely to convert"
```

---

## 🔄 Feature Mapping: HR → B2B Lead Scoring

LeadForge transforms HR features into lead scoring features:

| HR Feature | Maps To | Lead Interpretation |
|-----------|---------|------------------|
| Age | Company Maturity | Newer vs established companies |
| JobLevel | Decision Maker Seniority | Higher level = higher deal value |
| YearsAtCompany | Account Tenure | Longer relationship = better convert |
| MonthlyIncome | Deal Size Potential | Higher income = bigger deals |
| JobSatisfaction | Product Fit | Satisfaction = finding value |
| EnvironmentSatisfaction | Company Health | Healthy orgs buy more |
| Attrition | Conversion Probability | Binary conversion target |

---

## 🛠️ Data Preprocessing Pipeline

### Step 1: Load Data
```python
import pandas as pd

df = pd.read_csv('datasets/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(df.head())
print(df.info())
print(df['Attrition'].value_counts())
```

### Step 2: Clean Data
```python
# Check missing values
print(df.isnull().sum())  # Should be 0

# Check duplicates
print(df.duplicated().sum())

# Check data types
print(df.dtypes)
```

### Step 3: Encode Categorical Variables
```python
# Binary encoding
df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
df['OverTime'] = (df['OverTime'] == 'Yes').astype(int)

# One-hot encoding for multi-class
df = pd.get_dummies(df, columns=['Department', 'JobRole', 'MaritalStatus', 'Gender'])
```

### Step 4: Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
```

### Step 5: Train/Test Split
```python
from sklearn.model_selection import train_test_split

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train positive ratio: {y_train.mean():.2%}")
print(f"Test positive ratio: {y_test.mean():.2%}")
```

---

## 📊 Data Exploration Checklist

Before training, run these checks:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# 1. Basic Statistics
print(df.describe())
print(df.dtypes)

# 2. Class Distribution
print("\nTarget Distribution:")
print(df['Attrition'].value_counts())
print(f"Positive class ratio: {(df['Attrition'] == 'Yes').mean():.2%}")

# 3. Missing Values
print("\nMissing Values:")
print(df.isnull().sum().sum())  # Should be 0

# 4. Feature Distributions
df.select_dtypes(include=['int64', 'float64']).hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

# 5. Correlation Analysis
numeric_df = df.select_dtypes(include=['int64', 'float64'])
correlation = numeric_df.corr()
print("\nTop correlations with target (Attrition encoded):")
numeric_df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
print(numeric_df.corr()['Attrition'].sort_values(ascending=False).head(10))

# 6. Categorical Distribution
print("\nCategorical Features:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'Attrition':
        print(f"{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts())
```

---

## 🚀 Quick Start: Load Data in Code

### Using Preprocessing Script
```bash
cd datasets
python preprocessing.py --input raw/WA_Fn-UseC_-HR-Employee-Attrition.csv --output train_data.csv
```

### Manual in Python
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load
df = pd.read_csv('datasets/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Encode target
df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)

# One-hot encode categoricals
categorical_cols = ['Department', 'JobRole', 'MaritalStatus', 'Gender', 'OverTime']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Scale numeric features
numeric_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64'] and c != 'Attrition']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Attrition', axis=1),
    df['Attrition'],
    test_size=0.2,
    random_state=42,
    stratify=df['Attrition']
)

print(f"✅ Data ready! Train: {X_train.shape}, Test: {X_test.shape}")
```

---

## 🔄 Alternative Datasets

If you prefer different data:

### 1. LinkedIn Job Change Prediction
- **Kaggle URL**: https://www.kaggle.com/datasets/hasnainali/linkedin-job-change-prediction
- **Records**: 19,158
- **Features**: 14 (education, experience, demographics)
- **Target**: Binary (job change / stay)
- **Best for**: Career transition patterns as engagement proxy

### 2. Telco Customer Churn
- **Kaggle URL**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Records**: 7,043
- **Features**: 20 (service subscriptions, charges, tenure)
- **Target**: Binary (churned / retained)
- **Best for**: Service-based B2B models

### 3. Microsoft Salesforce Leads
- **Kaggle URL**: https://www.kaggle.com/datasets/yusheng9405/salesforce-leads
- **Records**: 50k+ actual leads
- **Features**: CRM-native (industry, lead source, status)
- **Target**: Binary (converted / not converted)
- **Best for**: Direct B2B application (most realistic)

---

## 🧬 Synthetic Data Generation

If you can't access Kaggle:

```python
# datasets/generate_synthetic_crm.py
import pandas as pd
import numpy as np

def generate_synthetic_crm(n_leads=50000, random_state=42):
    """Generate synthetic B2B CRM data"""
    np.random.seed(random_state)
    
    data = {
        'lead_id': [f'LEAD_{i:06d}' for i in range(n_leads)],
        'company_size': np.random.exponential(500, n_leads),
        'industry': np.random.choice(['SaaS', 'Finance', 'Healthcare', 'Retail', 'Tech'], n_leads),
        'email_opens': np.random.poisson(5, n_leads),
        'email_clicks': np.random.poisson(2, n_leads),
        'days_since_contact': np.random.poisson(15, n_leads),
        'deal_value': np.random.exponential(50000, n_leads),
        'engagement_score': np.random.uniform(0, 100, n_leads),
        'last_activity_type': np.random.choice(['email', 'call', 'meeting', 'demo'], n_leads),
        'converted': np.random.binomial(1, 0.25, n_leads)  # 25% conversion
    }
    
    df = pd.DataFrame(data)
    df.to_csv('train_data.csv', index=False)
    print(f"Generated {n_leads} synthetic leads")

if __name__ == '__main__':
    generate_synthetic_crm()
```

Run it:
```bash
python generate_synthetic_crm.py
```

---

## ✅ Verification Checklist

After downloading, verify your data:

- [ ] File exists: `datasets/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- [ ] File size: ~1.5 MB
- [ ] Rows: 1,470
- [ ] Columns: 35
- [ ] No missing values
- [ ] Attrition column exists
- [ ] Can load in pandas without errors

```bash
# Quick verification
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('datasets/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(f"✅ Rows: {len(df)}, Cols: {len(df.columns)}")
print(f"✅ Target distribution: {df['Attrition'].value_counts().to_dict()}")
print(f"✅ Missing values: {df.isnull().sum().sum()}")
EOF
```

---

## 📚 Documentation References

- Kaggle Dataset: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- Feature Descriptions: Included in dataset download (metadata.txt)
- Data Dictionary: Available on dataset page

---

## 🚨 Troubleshooting

**Issue: "No such file or directory" when loading data**
```bash
# Ensure file is in correct location
ls -la datasets/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv

# If not found, re-download from Kaggle
```

**Issue: Permission denied on Kaggle API**
```bash
# Fix permissions on kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Issue: "ModuleNotFoundError: No module named 'pandas'"**
```bash
# Install required libraries
pip install pandas scikit-learn xgboost
```

---

**Last Updated**: July 2025  
**Dataset Version**: 2024.1  
**Recommended for**: Novice to Advanced ML practitioners
