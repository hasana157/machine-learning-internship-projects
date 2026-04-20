"""
app.py — Interactive Streamlit application for Topic Modeling inference.

Run:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

# Make src importable when launching from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt  # Swapped Matplotlib for Altair for beautiful interactive charts

from src.models.predict import TopicPredictor
from src.utils import load_config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lumina Topic Explorer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a Premium Look ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Typography & Headers */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2E86AB, #562B82);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    /* Keyword Chips */
    .keyword-chip {
        display: inline-block;
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        color: #0284c7;
        border: 1px solid #bae6fd;
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    .keyword-chip:hover {
        background: #0284c7;
        color: #ffffff;
        transform: scale(1.05);
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP Engines…")
def get_predictor(model_file: str, vec_file: str) -> TopicPredictor:
    return TopicPredictor(
        model_filename=model_file,
        vectorizer_filename=vec_file,
        n_top_words=15,
    )

# ── Dynamic Topic Namer ───────────────────────────────────────────────────────
def generate_topic_name(keywords_str):
    """Dynamically creates a professional name based on top 2 keywords."""
    words = keywords_str.split(", ")
    if len(words) >= 2:
        return f"{words[0].title()} & {words[1].title()}"
    return "General Topic"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✨ Lumina Settings")
    st.write("Configure your analysis engine below.")

    model_choice = st.selectbox(
        "🧠 NLP Engine",
        options=["NMF (Non-negative Matrix Factorisation)",
                 "LDA (Latent Dirichlet Allocation)"],
        index=0,
    )
    model_file = (
        "nmf_model.joblib"
        if model_choice.startswith("NMF")
        else "lda_model.joblib"
    )
    vec_file = "vectorizer.joblib"

    n_display = st.slider("📊 Top topics to chart", min_value=3, max_value=20, value=5)

    st.markdown("---")
    st.markdown(
        "**Lumina Text Analytics v1.0** \n"
        "Unsupervised NLP with 20 Newsgroups  \n"
        "[View Documentation](#)"
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Lumina Topic Explorer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Paste your content below and let the AI instantly discover its latent thematic structures.</p>',
    unsafe_allow_html=True,
)

# ── Load model ────────────────────────────────────────────────────────────────
try:
    predictor = get_predictor(model_file, vec_file)
    # Silent success or a small toast looks more professional than a big green banner
    st.toast(f"Engine loaded: {type(predictor.model).__name__} | {predictor.n_topics} Topics Active", icon="✅")
except FileNotFoundError as exc:
    st.error(
        "⚠️ **Model files missing!** \n"
        "Please train the model first by running:  \n"
        "`python -m src.models.train`",
    )
    st.stop()

# ── Input area ────────────────────────────────────────────────────────────────
tab_input, tab_topics = st.tabs(["🔍 Intelligent Prediction", "📚 Topic Dictionary"])

with tab_input:
    example_texts = {
        "— Select a sample text —": "",
        "Space & Astronomy": (
            "NASA engineers have successfully tested the new ion thruster that "
            "will power the spacecraft on its journey to Mars. Scientists expect "
            "the mission to launch within the next 18 months."
        ),
        "Computer Hardware": (
            "The new GPU architecture delivers 40% faster rasterisation compared "
            "to the previous generation. Benchmark results show significant "
            "improvements in memory bandwidth and shader throughput."
        ),
        "Politics & Government": (
            "The senate approved a bipartisan infrastructure bill allocating "
            "funds for road repair, broadband expansion, and climate resilience "
            "projects across all 50 states."
        ),
        "Sports": (
            "The home team secured a dramatic playoff victory in overtime, "
            "with the star forward scoring three points in the final minute. "
            "The coach praised the team's defensive performance throughout."
        ),
    }

    selected = st.selectbox("Quick Load", list(example_texts.keys()))
    default_text = example_texts[selected]

    user_text = st.text_area(
        "Document Text",
        value=default_text,
        height=180,
        placeholder="Paste your article, email, or report here to analyze...",
        label_visibility="collapsed"
    )

    predict_btn = st.button("✨ Analyze Text", type="primary")

    if predict_btn:
        if not user_text.strip():
            st.warning("Please enter some text to begin analysis.")
        else:
            with st.spinner("Decoding latent topics..."):
                result = predictor.predict_single(user_text)

            # --- Derived Smart Labels ---
            dominant_kws = result["top_words"]
            smart_dominant_label = generate_topic_name(dominant_kws)
            dominant_weight = result["distribution"][result["dominant_topic"]]

            st.markdown("### 🏆 Primary Theme Discovered")
            
            # Using columns for a dashboard feel
            col1, col2, col3 = st.columns([1.5, 1, 2])
            with col1:
                st.metric("Detected Topic", smart_dominant_label)
            with col2:
                st.metric("Confidence Score", f"{dominant_weight * 100:.1f} %")
            with col3:
                # Elegant Keyword Chips
                keywords = dominant_kws.split(", ")
                chips_html = "".join(f'<span class="keyword-chip">{kw}</span>' for kw in keywords[:8])
                st.markdown("**Key Semantic Markers:**", unsafe_allow_html=True)
                st.markdown(chips_html, unsafe_allow_html=True)

            st.write("")
            st.write("")

            # ── Interactive Topic Weight Chart (Altair) ───────────────────────
            st.markdown(f"### 📊 Thematic Distribution (Top {n_display})")
            
            # Process dataframe for visualization
            dist_df = pd.DataFrame(result["topic_weights"])
            dist_df["Smart Label"] = dist_df["top_words"].apply(generate_topic_name)
            
            # Sort and slice
            plot_df = dist_df.head(n_display).copy()
            plot_df["Percentage"] = plot_df["weight"] * 100
            
            # Altair Chart (Highly Interactive and Clean)
            chart = alt.Chart(plot_df).mark_bar(cornerRadiusEnd=4, height=30).encode(
                x=alt.X('Percentage:Q', title='Confidence %', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('Smart Label:N', sort='-x', title='', axis=alt.Axis(labelLimit=200, labelFontSize=12)),
                color=alt.Color('Percentage:Q', scale=alt.Scale(scheme='blues'), legend=None),
                tooltip=[
                    alt.Tooltip('Smart Label:N', title='Topic'),
                    alt.Tooltip('Percentage:Q', title='Confidence', format='.1f'),
                    alt.Tooltip('top_words:N', title='Keywords')
                ]
            ).properties(height=max(250, n_display * 40))
            
            st.altair_chart(chart, use_container_width=True)

            st.divider()

            # ── Full distribution table ───────────────────────────────────────
            with st.expander("📋 View Complete Topic Breakdown"):
                dist_df["Weight"] = dist_df["weight"].map(lambda x: f"{x*100:.2f}%")
                st.dataframe(
                    dist_df[["Smart Label", "Weight", "top_words", "label"]].rename(
                        columns={"label": "System ID", "top_words": "Associated Keywords"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

with tab_topics:
    st.markdown("### 📚 Thematic Knowledge Base")
    st.markdown(
        "Explore all latent thematic clusters successfully mapped by the unsupervised engine. "
        "Each cluster represents a distinct semantic grouping."
    )

    topic_table = predictor.get_topic_table()

    # Apply smart naming to the browser table too
    for _, row in topic_table.iterrows():
        smart_name = generate_topic_name(row["top_words"])
        with st.expander(f"**{smart_name}** (ID: {row['label']})"):
            keywords = row["top_words"].split(", ")
            chips = "".join(f'<span class="keyword-chip">{kw}</span>' for kw in keywords)
            st.markdown(chips, unsafe_allow_html=True)

    st.divider()
    csv_bytes = topic_table.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Export Complete Dictionary (CSV)",
        data=csv_bytes,
        file_name="lumina_topics.csv",
        mime="text/csv",
        type="primary"
    )