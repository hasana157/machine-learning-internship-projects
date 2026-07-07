"""
Streamlit web app for the Cat vs Dog Transfer Learning Classifier.

Upload a photo of a cat or dog and see the model's prediction with a
confidence score.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Make "src" importable when this file is run directly by Streamlit
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import CLASS_NAMES, IMG_SIZE, MODEL_PATH

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    """Resize + batch only — MobileNetV2 preprocessing is baked into the model."""
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def main():
    st.title("🐱 vs 🐶 — Cat vs Dog Classifier")
    st.write(
        "This app uses a MobileNetV2-based transfer learning model "
        "(pretrained on ImageNet, fine-tuned on Cats vs Dogs) to classify "
        "uploaded photos."
    )

    model = load_model()
    if model is None:
        st.error(
            f"No trained model found at `{MODEL_PATH}`.\n\n"
            "Train one first by running:\n\n```\npython -m src.train\n```"
        )
        return

    uploaded_file = st.file_uploader(
        "Upload a photo of a cat or dog", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded image", use_container_width=True)

        if st.button("Predict"):
            img_array = preprocess(img)
            prob = float(model.predict(img_array, verbose=0)[0][0])  # P(dog)
            label = CLASS_NAMES[1] if prob >= 0.5 else CLASS_NAMES[0]
            confidence = prob if prob >= 0.5 else 1 - prob

            st.subheader(f"Prediction: **{label}** ({confidence:.1%} confidence)")

            st.write("Probability breakdown:")
            st.progress(prob if label == "dog" else 1 - prob)
            col1, col2 = st.columns(2)
            col1.metric("P(cat)", f"{1 - prob:.1%}")
            col2.metric("P(dog)", f"{prob:.1%}")

    st.divider()
    st.caption(
        "Model: MobileNetV2 backbone + custom head, frozen then fine-tuned · "
        "Project 12 — Cat vs Dog Classifier"
    )


if __name__ == "__main__":
    main()
