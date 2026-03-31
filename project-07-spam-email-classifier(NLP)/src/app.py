# advanced_app_v2.py

import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import os

# -----------------------------
# Page Config & Theme
# -----------------------------
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for dark / detective style
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 40px;
    }
    .stTextArea>div>div>textarea {
        background-color: #1c1c1c;
        color: white;
    }
    .stProgress>div>div>div>div {
        background-color: #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "../models/spam_pipeline.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# -----------------------------
# Preprocessing
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------
# Prediction Function
# -----------------------------
def predict(text):
    clean = clean_text(text)
    pred = model.predict([clean])[0]
    prob = model.predict_proba([clean])[0]
    label = "Spam 🚨" if pred == 1 else "Not Spam ✅"
    confidence = np.max(prob)

    # Top contributing words
    tfidf = model.named_steps['tfidf']
    clf = model.named_steps['clf']
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]

    X_vect = tfidf.transform([clean]).toarray()[0]
    contrib = X_vect * coefs
    top_idx = np.argsort(np.abs(contrib))[-5:][::-1]
    top_words = [(feature_names[i], contrib[i]) for i in top_idx if contrib[i] != 0]

    return label, confidence, top_words

# -----------------------------
# Sidebar / Info
# -----------------------------
st.sidebar.title("📩 Spam Email Classifier")
st.sidebar.info(
    """
    - Paste your email or SMS message for detection.
    - Upload a CSV for batch prediction (`text` column required).
    - Model: TF-IDF + Logistic Regression.
    """
)

# -----------------------------
# UI Layout
# -----------------------------
st.title("📩 Spam Email Classifier ")
st.markdown("**Real-time detection, batch upload & probability visualization**")

# Text input
user_input = st.text_area(
    "Type or paste your message here:",
    height=150,
    placeholder="Enter an email or SMS message..."
)

# Buttons
col1, col2 = st.columns(2)
predict_btn = col1.button("Predict")
reset_btn = col2.button("Reset")

# Handle Reset
if reset_btn:
    user_input = ""
    st.experimental_rerun()

# Prediction
if predict_btn and user_input.strip():
    label, confidence, top_words = predict(user_input)
    if "Spam" in label:
        st.error(f"🚨 Prediction: {label}")
    else:
        st.success(f"✅ Prediction: {label}")

    st.write(f"**Confidence:** {confidence:.2f}")
    st.progress(confidence)

    st.markdown("### 🔑 Top Contributing Words")
    for word, score in top_words:
        st.write(f"- **{word}** ({score:.2f})")

# -----------------------------
# Batch Upload
# -----------------------------
st.markdown("---")
st.markdown("### Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload CSV file with a 'text' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column")
    else:
        results = []
        for msg in df['text']:
            label, confidence, _ = predict(msg)
            results.append([msg, label, confidence])
        res_df = pd.DataFrame(results, columns=['Message', 'Prediction', 'Confidence'])
        st.write(res_df)
        st.download_button(
            label="Download Predictions CSV",
            data=res_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Built with ❤️ using TF-IDF + Logistic Regression")