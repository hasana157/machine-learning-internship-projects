"""
Streamlit web app for the MNIST Handwritten Digit Classifier.

Lets a user either:
  - draw a digit on a canvas, or
  - upload an image of a handwritten digit
and see the model's prediction with a confidence bar chart.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# Make "src" importable when this file is run directly by Streamlit
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import MODEL_PATH
from src.preprocessing import preprocess_pil_image

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def show_prediction(img_array: np.ndarray, model):
    pred = model.predict(img_array, verbose=0)[0]
    digit = int(pred.argmax())
    confidence = float(pred.max())

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            (img_array.reshape(28, 28) * 255).astype("uint8"),
            caption="Model input (28x28)",
            width=150,
        )
    with col2:
        st.metric("Predicted digit", digit)
        st.metric("Confidence", f"{confidence:.1%}")

    st.subheader("Confidence by digit")
    df = pd.DataFrame({"digit": list(range(10)), "confidence": pred})
    st.bar_chart(df.set_index("digit"))


def main():
    st.title("🔢 Handwritten Digit Classifier")
    st.write(
        "This app uses a neural network trained on MNIST to recognize "
        "handwritten digits (0-9). Draw a digit or upload an image."
    )

    model = load_model()
    if model is None:
        st.error(
            f"No trained model found at `{MODEL_PATH}`.\n\n"
            "Train one first by running:\n\n```\npython -m src.train\n```"
        )
        return

    tab_draw, tab_upload = st.tabs(["✏️ Draw a digit", "📁 Upload an image"])

    with tab_draw:
        try:
            from streamlit_drawable_canvas import st_canvas
        except ImportError:
            st.warning(
                "Drawing requires the `streamlit-drawable-canvas` package.\n\n"
                "Install it with:\n```\npip install streamlit-drawable-canvas\n```"
            )
        else:
            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=18,
                stroke_color="white",
                background_color="black",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="canvas",
            )

            if st.button("Predict drawn digit"):
                if canvas_result.image_data is not None:
                    img = Image.fromarray(
                        canvas_result.image_data.astype("uint8")
                    ).convert("RGB")
                    img_array = preprocess_pil_image(img, auto_invert=False)
                    show_prediction(img_array, model)
                else:
                    st.warning("Draw a digit first!")

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a handwritten digit image", type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded image", width=150)
            if st.button("Predict uploaded digit"):
                img_array = preprocess_pil_image(img, auto_invert=True)
                show_prediction(img_array, model)

    st.divider()
    st.caption(
        "Model: Fully connected neural network trained on MNIST · "
        "Project 14 — Handwritten Digit Classifier"
    )


if __name__ == "__main__":
    main()
