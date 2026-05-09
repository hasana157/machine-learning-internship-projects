"""
streamlit_app.py
----------------
Streamlit web application for CIFAR-10 image classification.

Run with:
    streamlit run app/streamlit_app.py

Features:
    - Upload any image (JPG, PNG, WEBP)
    - Select baseline or transfer learning model
    - View top-3 predictions with confidence bars
    - Toggle Grad-CAM overlay for model explainability
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path when running from app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #667eea;
    }
    .top-pred {
        border-left: 4px solid #28a745;
        background: #f0fff4;
    }
    .confidence-label {
        font-size: 0.85rem;
        color: #555;
    }
    .class-badge {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2d3748;
    }
    .info-box {
        background: #e8f4fd;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.9rem;
        color: #2c5282;
    }
</style>
""", unsafe_allow_html=True)

# ── Import project modules ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_cached_model(model_type: str):
    from src.inference import load_model
    return load_model(model_type)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    model_choice = st.selectbox(
        "Model",
        options=["transfer", "baseline"],
        format_func=lambda x: "🚀 MobileNetV2 (Transfer)" if x == "transfer" else "🏗️ Baseline CNN",
        help="Transfer model is more accurate; baseline is faster.",
    )

    show_gradcam = st.checkbox(
        "Show Grad-CAM",
        value=False,
        help="Highlight image regions that influenced the prediction.",
    )

    top_k = st.slider("Show top-K predictions", min_value=1, max_value=10, value=3)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
**CIFAR-10** contains 60 000 images across 10 classes:

airplane · automobile · bird · cat · deer ·
dog · frog · horse · ship · truck

Both models were trained with:
- Data augmentation
- Early stopping
- Learning rate scheduling
""")

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">CIFAR-10 Image Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload an image → get instant predictions with confidence scores</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown(
        """
        <div class="info-box">
        👆 <strong>Upload any image</strong> to classify it using a CNN trained on CIFAR-10.<br><br>
        Best results: images containing one of the 10 CIFAR-10 categories
        (vehicles, animals, aircraft, ships).
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Load image ────────────────────────────────────────────────────────────────
pil_image = Image.open(uploaded).convert("RGB")

col_img, col_results = st.columns([1, 1.4], gap="large")

with col_img:
    st.image(pil_image, caption="Uploaded image", use_column_width=True)

# ── Load model & predict ──────────────────────────────────────────────────────
try:
    model = load_cached_model(model_choice)
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python train.py` first to train and save the models.")
    st.stop()

from src.inference import predict, preprocess_image

with st.spinner("Classifying…"):
    results = predict(model, pil_image, model_type=model_choice, top_k=top_k)

# ── Display results ───────────────────────────────────────────────────────────
with col_results:
    st.markdown("### 🎯 Predictions")

    for rank, r in enumerate(results):
        css_class = "prediction-card top-pred" if rank == 0 else "prediction-card"
        icon = "🥇" if rank == 0 else ("🥈" if rank == 1 else "🥉")
        pct = r["confidence"] * 100

        st.markdown(
            f"""
            <div class="{css_class}">
                <span class="class-badge">{icon} {r['class'].capitalize()}</span>
                <span class="confidence-label" style="float:right">{pct:.1f}%</span>
                <br>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(r["confidence"])

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
if show_gradcam:
    st.markdown("---")
    st.markdown("### 🔥 Grad-CAM Explanation")
    st.caption(
        "Highlighted regions had the greatest influence on the top prediction. "
        "Red = high importance, Blue = low importance."
    )

    try:
        import cv2
        from utils.gradcam import compute_gradcam, overlay_gradcam
        from src.config import IMAGE_SIZE_TL, IMAGE_SIZE

        target = IMAGE_SIZE_TL if model_choice == "transfer" else IMAGE_SIZE
        x_proc = preprocess_image(pil_image, model_type=model_choice)

        # Grad-CAM needs a Conv2D layer; find it automatically
        conv_name = None
        for layer in reversed(model.layers):
            import tensorflow.keras as keras
            if isinstance(layer, keras.layers.Conv2D):
                conv_name = layer.name
                break

        if conv_name:
            heatmap = compute_gradcam(model, x_proc, conv_layer_name=conv_name)

            # Display image resized to match model input
            img_resized = np.array(pil_image.resize(target, Image.BILINEAR))
            overlay = overlay_gradcam(img_resized, heatmap)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_resized);           axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(heatmap, cmap="jet");   axes[1].set_title("Heatmap"); axes[1].axis("off")
            axes[2].imshow(overlay);               axes[2].set_title("Overlay"); axes[2].axis("off")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No Conv2D layer found for Grad-CAM in this model.")

    except Exception as e:
        st.warning(f"Grad-CAM unavailable: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with TensorFlow · MobileNetV2 · Streamlit · CIFAR-10  |  "
    "Models trained with EarlyStopping + ReduceLROnPlateau"
)
