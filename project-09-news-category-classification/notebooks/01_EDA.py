"""
notebooks/01_EDA.py
===================
Exploratory Data Analysis — AG News Dataset.
Run standalone: python notebooks/01_EDA.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader  import load_from_csv
from src.preprocessor import clean_dataframe

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

train_df = load_from_csv("data/raw/ag_train.csv")
test_df  = load_from_csv("data/raw/ag_test.csv")
print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
print(f"\nLabel distribution:\n{train_df['label'].value_counts()}")

# Sample texts
print("\n=== Sample texts per class ===")
for cls in CLASS_NAMES:
    sample = train_df[train_df["label"]==cls].sample(1, random_state=42)["text"].values[0]
    print(f"\n[{cls}]\n{sample[:200]}…")

# Text length analysis
train_df = clean_dataframe(train_df, "text")
train_df["word_count"] = train_df["text_clean"].str.split().str.len()
print("\n=== Word Count Statistics ===")
print(train_df.groupby("label")["word_count"].describe().round(1))

# Plot length distribution
PALETTE = ["#2563EB","#16A34A","#DC2626","#D97706"]
Path("reports/figures").mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(9,4))
for cls, color in zip(CLASS_NAMES, PALETTE):
    data = train_df[train_df["label"]==cls]["word_count"]
    ax.hist(data, bins=40, alpha=0.6, color=color, label=cls, density=True)
ax.set_xlabel("Word Count"); ax.set_ylabel("Density")
ax.set_title("Text Length Distribution by Category", fontsize=14, fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig("reports/figures/text_length_distribution.png", dpi=150, bbox_inches="tight")
print("\nSaved → reports/figures/text_length_distribution.png")
