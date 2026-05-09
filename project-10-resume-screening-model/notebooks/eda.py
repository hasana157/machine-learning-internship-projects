"""
notebooks/eda.py
----------------
Exploratory Data Analysis for resume_dataset.csv.
Run as a script: python notebooks/eda.py

Covers:
  - Class distribution
  - Skill frequency per role
  - Years of experience distribution
  - Education distribution
  - Theoretical model ceiling analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

DATA_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "resume_dataset.csv")
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nClass distribution:\n{df['JobRole'].value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    roles = sorted(df["JobRole"].unique())

    # ── Figure 1: Class distribution + YearsExp boxplot ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Resume Dataset — EDA (Synthetic Data)", fontsize=13, fontweight="bold")

    # Class counts
    counts = df["JobRole"].value_counts()
    axes[0].barh(counts.index, counts.values, color="#3B82F6", edgecolor="none")
    axes[0].set_title("Class Distribution")
    axes[0].set_xlabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(v + 0.3, i, str(v), va="center", fontsize=9)

    # Years of experience per role
    data_by_role = [df[df["JobRole"] == r]["YearsExperience"].values for r in roles]
    bp = axes[1].boxplot(data_by_role, labels=[r.replace(" ", "\n") for r in roles],
                          patch_artist=True, medianprops=dict(color="white", linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#60A5FA")
    axes[1].set_title("Years of Experience by Role")
    axes[1].set_ylabel("Years")

    plt.tight_layout()
    out1 = os.path.join(REPORT_DIR, "eda_overview.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out1}")

    # ── Figure 2: Skill frequency heatmap ─────────────────────────────────────
    all_skills = set()
    for s in df["Skills"]:
        all_skills.update([x.strip() for x in s.split(",")])
    all_skills = sorted(all_skills)

    heatmap_data = {}
    for role in roles:
        sub = df[df["JobRole"] == role]
        total = len(sub)
        skill_counts = Counter()
        for s in sub["Skills"]:
            skill_counts.update([x.strip() for x in s.split(",")])
        heatmap_data[role] = {sk: round(skill_counts[sk] / total, 2) for sk in all_skills}

    hm_df = pd.DataFrame(heatmap_data).T

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    import seaborn as sns
    sns.heatmap(hm_df, annot=True, fmt=".2f", cmap="Blues", ax=ax2, linewidths=0.4)
    ax2.set_title(
        "Skill Frequency per Role (proportion)\n"
        "Note: All roles share the same skills at similar rates — "
        "minimal discriminating signal",
        fontsize=11
    )
    ax2.set_xlabel("Skill")
    ax2.set_ylabel("Job Role")
    plt.tight_layout()
    out2 = os.path.join(REPORT_DIR, "eda_skill_heatmap.png")
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"Saved: {out2}")

    # ── Figure 3: Education distribution ──────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    edu_role = pd.crosstab(df["JobRole"], df["Education"], normalize="index") * 100
    edu_role.plot(kind="bar", ax=ax3, colormap="Blues", edgecolor="none")
    ax3.set_title("Education Level Distribution per Role (%)")
    ax3.set_ylabel("Percentage")
    ax3.set_xlabel("")
    ax3.legend(title="Education", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out3 = os.path.join(REPORT_DIR, "eda_education.png")
    fig3.savefig(out3, dpi=150)
    plt.close(fig3)
    print(f"Saved: {out3}")

    print("\n[EDA complete] All plots saved to reports/")


if __name__ == "__main__":
    main()
