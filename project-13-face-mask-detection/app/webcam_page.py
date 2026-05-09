"""
webcam_page.py — Streamlit continuous webcam component.

Imported by app.py to render the live webcam tab.
Uses st.camera_input() for browser-based capture (one frame at a time)
because browsers block direct camera access from Python.

For TRUE real-time video, run:
    python webcam_detect.py
That opens a native OpenCV window with live video — no frame limit.
"""

import os, sys, time
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def render_webcam_tab(pipeline, demo_predict_fn, det_conf: float):
    """Render the webcam detection tab inside Streamlit."""

    st.subheader("Webcam Detection")

    col_info, col_cam = st.columns([1, 2])

    with col_info:
        st.markdown("""
#### Two ways to use your webcam

**Option A — Browser photo (this page)**  
Click *Take photo* to capture a single frame and run the full
detection pipeline on it. Works in any browser, no install needed.

**Option B — Real-time video (recommended)**  
Run the standalone script for live video at 30 fps with a native
OpenCV window:

```bash
# Auto-detects model and camera
python webcam_detect.py

# Explicit settings
python webcam_detect.py \\
    --camera 0 \\
    --model models/mask_classifier.keras \\
    --det_conf 0.45

# No model needed (heuristic mode)
python webcam_detect.py --no_model
```

**Keys inside the OpenCV window:**
| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `S` | Save screenshot |
| `P` | Pause / resume |
        """)

        st.info(
            "The Streamlit browser tab captures **one frame at a time** "
            "due to browser security restrictions. For continuous live detection "
            "use `webcam_detect.py`.",
            icon="ℹ️",
        )

    with col_cam:
        st.markdown("**Capture a photo from your webcam:**")
        img_data = st.camera_input(
            label="Camera",
            label_visibility="collapsed",
            key="webcam_capture",
        )

        if img_data is not None:
            pil_img = Image.open(img_data)
            img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

            t0 = time.perf_counter()
            with st.spinner("Detecting…"):
                if pipeline is None:
                    annotated, dets = demo_predict_fn(img_bgr)
                else:
                    annotated_raw, preds = pipeline.predict_image(img_bgr, det_conf)
                    annotated = annotated_raw
                    dets = [p.to_dict() for p in preds]
            elapsed = time.perf_counter() - t0

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True,
                     caption=f"Inference: {elapsed*1000:.0f} ms  |  {len(dets)} face(s)")

            if dets:
                n_mask   = sum(1 for d in dets if d["label"] == "Mask")
                n_nomask = len(dets) - n_mask
                m1, m2, m3 = st.columns(3)
                m1.metric("Faces", len(dets))
                m2.metric("Masked",   n_mask,   delta=None)
                m3.metric("No Mask",  n_nomask, delta=None)

                for i, d in enumerate(dets):
                    color = "#00c853" if d["label"] == "Mask" else "#f44336"
                    pill  = "pill-mask" if d["label"] == "Mask" else "pill-nomask"
                    st.markdown(
                        f'<div style="margin:.3rem 0;">'
                        f'<span class="{pill}">{d["label"]}</span>'
                        f'&nbsp; Face {i+1} &nbsp;|&nbsp; '
                        f'<strong>{d["confidence"]:.1%}</strong> confidence'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("No faces detected — try better lighting or move closer.")
