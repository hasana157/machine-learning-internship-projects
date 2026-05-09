"""
app/streamlit_app.py
--------------------
Streamlit web UI for the Resume Screening classification demo.

Run:
    streamlit run app/streamlit_app.py

WARNING: EDUCATIONAL / DEMO PURPOSES ONLY.
         Must NOT be used in any real hiring workflow.
"""

import os
import sys
import json
import logging

# Ensure project root is on path when launched from /app directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import streamlit as st

from src.predictor import ResumePredictor
from src.config import METRICS_PATH

logging.basicConfig(level=logging.WARNING)

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Classifier | Project 10",
    page_icon="📄",
    layout="centered",
)

# ─── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def get_predictor() -> ResumePredictor:
    return ResumePredictor()


@st.cache_data
def load_metrics() -> dict | None:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


# ─── Header ────────────────────────────────────────────────────────────────────
st.title("📄 Resume Screening Classifier")
st.caption("Project 10 — NLP Multi-Class Classification | Educational Demo")

st.error(
    "**Responsible AI Notice** — This tool is for **educational / demo purposes "
    "only**. It must NOT be used to make real hiring decisions, screen or reject "
    "candidates, or automate any part of a recruitment pipeline. Model predictions "
    "may reflect biases present in the synthetic training data.",
    icon="🚫",
)

# ─── Sidebar — Model Performance ───────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")

    metrics = load_metrics()
    if metrics:
        col1, col2 = st.columns(2)
        col1.metric("Macro F1",  f"{metrics['macro_f1']:.3f}")
        col2.metric("Accuracy",  f"{metrics['accuracy']:.3f}")
        col1.metric("Precision", f"{metrics['macro_precision']:.3f}")
        col2.metric("Recall",    f"{metrics['macro_recall']:.3f}")
        st.caption(f"Val samples: {metrics.get('num_val_samples', '—')}")

        st.markdown("---")
        st.subheader("Per-Class F1")
        if "per_class" in metrics:
            for cls, vals in sorted(metrics["per_class"].items()):
                f1 = vals["f1_score"]
                st.progress(
                    min(f1, 1.0),
                    text=f"{cls}: {f1:.2f} (n={vals['support']})"
                )

        st.markdown("---")
        with st.expander("Why are metrics low?"):
            st.markdown(
                "This dataset is **fully synthetic** — all 5 roles share the "
                "exact same 10 skills at similar frequencies. The theoretical "
                "ceiling (Random Forest CV) is ~0.22 macro F1, barely above "
                "random chance (0.20 for 5 classes).\n\n"
                "This is a **dataset limitation**, not a pipeline deficiency. "
                "The architecture is production-quality. Low metrics are "
                "reported honestly — inflating them would be irresponsible."
            )
    else:
        st.info("No metrics found. Run `python train.py` first.")

    st.markdown("---")
    st.caption("Model: Logistic Regression\nFeatures: TF-IDF + Numeric\nFramework: scikit-learn")

# ─── Main — Prediction ─────────────────────────────────────────────────────────
st.subheader("🔍 Classify a Resume")

st.markdown(
    "Enter skills, education level, and years of experience below. "
    "The model will predict the most likely job category."
)

PLACEHOLDER = (
    "Example:\n"
    "Python, Machine Learning, SQL, Deep Learning, Data Analysis\n"
    "Education: Masters\n"
    "Experience: 5 years"
)

resume_text = st.text_area(
    label="Resume Text Input",
    placeholder=PLACEHOLDER,
    height=180,
    label_visibility="collapsed",
)

predict_btn = st.button(
    "🚀  Predict Job Role",
    type="primary",
    use_container_width=True,
)

if predict_btn:
    if not resume_text.strip():
        st.warning("Please enter some resume text before predicting.")
    else:
        with st.spinner("Classifying..."):
            try:
                predictor = get_predictor()
                result = predictor.predict(resume_text)
            except FileNotFoundError:
                st.error(
                    "Model file not found. Please run `python train.py` first "
                    "to train and save the model."
                )
                st.stop()

        # ── Result display ─────────────────────────────────────────────────────
        st.divider()

        res_col, conf_col = st.columns([2, 1])
        with res_col:
            st.success(f"**Predicted Role:** {result['predicted_label']}")
        with conf_col:
            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

        # ── Confidence bar chart ───────────────────────────────────────────────
        st.subheader("Confidence Scores — All Classes")
        scores  = result["all_scores"]
        classes = list(scores.keys())
        values  = [scores[c] * 100 for c in classes]

        # Sort by confidence descending
        pairs = sorted(zip(classes, values), key=lambda x: -x[1])
        classes_sorted = [p[0] for p in pairs]
        values_sorted  = [p[1] for p in pairs]

        fig, ax = plt.subplots(figsize=(7, 3.2))
        colors = [
            "#1D4ED8" if c == result["predicted_label"] else "#93C5FD"
            for c in classes_sorted
        ]
        bars = ax.barh(
            classes_sorted[::-1],
            values_sorted[::-1],
            color=colors[::-1],
            height=0.5,
            edgecolor="none",
        )
        ax.set_xlabel("Confidence (%)", fontsize=10)
        ax.set_xlim(0, 108)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.tick_params(axis="x", labelsize=9)

        for bar, val in zip(bars, values_sorted[::-1]):
            ax.text(
                bar.get_width() + 1.2,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center", fontsize=9, color="#374151"
            )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Raw JSON ───────────────────────────────────────────────────────────
        with st.expander("Raw prediction output (JSON)"):
            st.json(result)

        st.info(
            "**Note:** Confidence scores on this synthetic dataset are "
            "near-uniform (~20% each) due to the dataset's lack of "
            "discriminating signal. This is expected and reported honestly.",
            icon="ℹ️",
        )

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Project 10 — Resume Screening Model  |  "
    "scikit-learn + Streamlit  |  "
    "Educational NLP Demo — NOT a Hiring Tool"
)
