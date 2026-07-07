"""
streamlit_app.py
================
VisualSentry — AI-Powered Visual Defect Detection
Professional Streamlit GUI with dark industrial theme.

Run with:
    streamlit run app/streamlit_app.py
"""

import io
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Ensure project root is on sys.path when running from app/ directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VisualSentry | AI Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — dark industrial theme
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

/* ── Root variables ───────────────────────────────────────────────────────── */
:root {
    --bg:       #0A1628;
    --surface:  #0F2040;
    --surface2: #142952;
    --accent:   #00C9A7;
    --accent2:  #00A8E8;
    --danger:   #FF4B4B;
    --success:  #2ECC71;
    --warning:  #F39C12;
    --text:     #D6E4F0;
    --muted:    #5E7A96;
    --border:   #1E3A5A;
    --mono:     'Share Tech Mono', monospace;
    --sans:     'Exo 2', sans-serif;
}

/* ── Global resets ────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--sans);
    color: var(--text);
}

.stApp {
    background: radial-gradient(ellipse at 20% 20%, #0D2040 0%, #050E1A 60%);
    background-attachment: fixed;
}

/* ── Remove Streamlit default padding ────────────────────────────────────── */
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }
header[data-testid="stHeader"] { background: rgba(10,22,40,0.95) !important; border-bottom: 1px solid var(--border); }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #091525 0%, #0A1E35 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .sidebar-logo {
    font-family: var(--mono);
    font-size: 1.5rem;
    color: var(--accent);
    text-shadow: 0 0 18px rgba(0,201,167,0.5);
    letter-spacing: 0.04em;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--accent);
}

/* ── Metric cards ─────────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,201,167,0.15); }
.metric-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); margin-bottom: 0.5rem; font-family: var(--mono); }
.metric-value { font-size: 2rem; font-weight: 700; color: var(--accent); font-family: var(--mono); text-shadow: 0 0 12px rgba(0,201,167,0.3); }
.metric-delta { font-size: 0.78rem; color: var(--muted); margin-top: 0.25rem; }

/* ── Status badges ────────────────────────────────────────────────────────── */
.badge-pass {
    display: inline-block; padding: 0.5rem 1.4rem; border-radius: 50px;
    background: linear-gradient(135deg, #1A4A2E, #2ECC71);
    color: #fff; font-weight: 700; font-size: 1.1rem;
    border: 1px solid var(--success); font-family: var(--mono);
    letter-spacing: 0.1em; box-shadow: 0 4px 15px rgba(46,204,113,0.3);
}
.badge-fail {
    display: inline-block; padding: 0.5rem 1.4rem; border-radius: 50px;
    background: linear-gradient(135deg, #4A1A1A, #FF4B4B);
    color: #fff; font-weight: 700; font-size: 1.1rem;
    border: 1px solid var(--danger); font-family: var(--mono);
    letter-spacing: 0.1em; box-shadow: 0 4px 15px rgba(255,75,75,0.3);
}
.badge-info {
    display: inline-block; padding: 0.3rem 0.9rem; border-radius: 6px;
    background: rgba(0,201,167,0.12); color: var(--accent);
    border: 1px solid rgba(0,201,167,0.3); font-size: 0.8rem;
    font-family: var(--mono); margin: 0.15rem;
}

/* ── Section headers ──────────────────────────────────────────────────────── */
.section-header {
    font-size: 1.0rem; font-family: var(--mono);
    color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.15em; margin-bottom: 1rem;
    border-bottom: 1px solid var(--border); padding-bottom: 0.5rem;
}
.hero-title {
    font-size: 2.8rem; font-weight: 700; font-family: var(--mono);
    background: linear-gradient(135deg, #00C9A7, #00A8E8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.0rem; color: var(--muted); letter-spacing: 0.06em;
    font-family: var(--mono); margin-top: 0.4rem;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #004D3F, #00C9A7) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-family: var(--mono) !important;
    font-size: 0.85rem !important; letter-spacing: 0.08em !important;
    transition: all 0.25s !important; text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,201,167,0.4) !important;
}

/* ── Inputs ───────────────────────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: var(--surface) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
    font-family: var(--mono) !important;
}
.stSlider [data-baseweb="slider"] { color: var(--accent); }

/* ── Dataframes ───────────────────────────────────────────────────────────── */
.dataframe { background: var(--surface) !important; color: var(--text) !important; }

/* ── Code blocks ──────────────────────────────────────────────────────────── */
code { background: #091525 !important; color: var(--accent) !important; font-family: var(--mono) !important; }
pre code { color: var(--text) !important; }

/* ── Divider ──────────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Tech stack pills ─────────────────────────────────────────────────────── */
.tech-pill {
    display: inline-block; padding: 0.25rem 0.8rem;
    border-radius: 20px; margin: 0.2rem;
    font-size: 0.78rem; font-family: var(--mono);
    font-weight: 600; letter-spacing: 0.05em;
}
.pill-tf   { background: rgba(255,111,0,0.15); color: #FF6F00; border: 1px solid rgba(255,111,0,0.3); }
.pill-st   { background: rgba(255,75,75,0.12);  color: #FF4B4B; border: 1px solid rgba(255,75,75,0.3); }
.pill-py   { background: rgba(0,168,232,0.12);  color: #00A8E8; border: 1px solid rgba(0,168,232,0.3); }
.pill-sk   { background: rgba(243,156,18,0.12); color: #F39C12; border: 1px solid rgba(243,156,18,0.3); }
.pill-np   { background: rgba(0,201,167,0.12);  color: #00C9A7; border: 1px solid rgba(0,201,167,0.3); }

/* ── Image containers ─────────────────────────────────────────────────────── */
.img-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.75rem; text-align: center;
}
.img-label { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }

/* ── Sidebar status badge ─────────────────────────────────────────────────── */
.status-ok  { color: var(--success); font-family: var(--mono); font-size: 0.82rem; }
.status-err { color: var(--danger);  font-family: var(--mono); font-size: 0.82rem; }

/* ── Scan line effect on hero ─────────────────────────────────────────────── */
.hero-section {
    position: relative;
    padding: 2rem 0 1.5rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — lazy imports & model caching
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str):
    """Load and cache the Keras autoencoder model.

    Args:
        model_path: Path to the saved .h5 model file.

    Returns:
        Loaded Keras model, or None if not found.
    """
    import tensorflow as tf
    if not Path(model_path).exists():
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception:
        return None


def get_config() -> dict:
    """Load configuration, defaulting gracefully on error.

    Returns:
        Config dict or empty dict fallback.
    """
    try:
        from src.utils import load_config
        return load_config("config.yaml")
    except Exception:
        return {}


def get_evaluator(model, cfg: dict):
    """Instantiate AnomalyEvaluator if model is available.

    Args:
        model: Loaded Keras model or None.
        cfg: Configuration dictionary.

    Returns:
        AnomalyEvaluator instance, or None.
    """
    if model is None:
        return None
    try:
        from src.evaluator import AnomalyEvaluator
        return AnomalyEvaluator(model=model, config_path="config.yaml")
    except Exception:
        return None


def get_last_trained_time(model_path: str) -> str:
    """Return human-readable modification time of the model file.

    Args:
        model_path: Path to the model file.

    Returns:
        Formatted datetime string or '—' if not found.
    """
    p = Path(model_path)
    if not p.exists():
        return "—"
    mtime = p.stat().st_mtime
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")


def score_uploaded_image(
    file_bytes: bytes,
    evaluator,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Process an uploaded image and return detection results.

    Args:
        file_bytes: Raw bytes of the uploaded image.
        evaluator: AnomalyEvaluator instance.
        cfg: Configuration dict.

    Returns:
        Tuple (original_arr, reconstruction_arr, heatmap_arr, score, is_defect).
    """
    from src.utils import bytes_to_numpy

    img_size = tuple(cfg["model"]["img_size"])
    img = bytes_to_numpy(file_bytes, img_size)

    inp = img[np.newaxis, ...]
    rec = evaluator.model.predict(inp, verbose=0)[0]

    orig_u8, overlay_u8, score = evaluator.generate_heatmap(img)
    rec_u8 = (np.clip(rec, 0, 1) * 255).astype(np.uint8)

    is_defect = score > evaluator.threshold
    return orig_u8, rec_u8, overlay_u8, score, is_defect


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

if "session_results" not in st.session_state:
    st.session_state["session_results"] = []

if "threshold" not in st.session_state:
    st.session_state["threshold"] = None

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

cfg = get_config()
model_path = cfg.get("paths", {}).get("model_save", "models/autoencoder_defect.h5")
model = load_model_cached(model_path)
evaluator = get_evaluator(model, cfg)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🔬 VisualSentry</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#4A6070;font-family:\'Share Tech Mono\',monospace;margin-top:-6px;margin-bottom:16px;">Detect what the human eye misses.</div>', unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Upload & Detect", "🧠 Model Training", "📈 Analytics", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()

    # Model status
    if model is not None:
        st.markdown('<div class="status-ok">✅ Model Loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-err">⚠️ No Model Found</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.72rem;color:#5E7A96;font-family:\'Share Tech Mono\',monospace;">Go to Model Training →</div>', unsafe_allow_html=True)

    st.markdown("")

    try:
        import tensorflow as tf
        tf_ver = tf.__version__
    except ImportError:
        tf_ver = "not installed"

    st.markdown(
        f"""
        <div style='font-size:0.72rem;color:#4A6070;font-family:"Share Tech Mono",monospace;line-height:1.8;'>
        TF&nbsp;&nbsp;&nbsp;&nbsp;: {tf_ver}<br>
        Model&nbsp;: {Path(model_path).name}<br>
        Trained: {get_last_trained_time(model_path)}<br>
        Lat.dim: {cfg.get('model',{{}}).get('latent_dim','—')}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_dashboard():
    """Render the main dashboard overview page."""
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-title">Visual Defect Detection</div>
            <div class="hero-subtitle">// AI-POWERED MANUFACTURING QUALITY CONTROL SYSTEM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results: List[dict] = st.session_state["session_results"]

    total = len(results)
    defects = sum(1 for r in results if r.get("is_defect"))
    rate = (defects / total * 100) if total else 0
    avg_score = np.mean([r["score"] for r in results]) if results else 0.0

    c1, c2, c3, c4 = st.columns(4)
    from src.utils import format_metric_card

    c1.markdown(format_metric_card("Total Analyzed", str(total), "This session"), unsafe_allow_html=True)
    c2.markdown(format_metric_card("Defects Detected", str(defects), f"{rate:.1f}% of total"), unsafe_allow_html=True)
    c3.markdown(format_metric_card("Detection Rate", f"{rate:.1f}%", "Threshold-based"), unsafe_allow_html=True)
    c4.markdown(format_metric_card("Avg Anomaly Score", f"{avg_score:.4f}", "Lower = more normal"), unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">Score Distribution</div>', unsafe_allow_html=True)
        if results:
            from src.utils import plot_score_bar
            df = pd.DataFrame(results)
            fig = plot_score_bar(df.rename(columns={"is_defect": "_is_defect"}), title="")

            scores = np.array([r["score"] for r in results])
            preds = ["fail" if r["is_defect"] else "pass" for r in results]
            df_plot = pd.DataFrame({"anomaly_score": scores, "predicted": preds})
            from src.utils import plot_score_bar, PLOTLY_LAYOUT
            fig2 = go.Figure(
                go.Bar(
                    x=list(range(len(df_plot))),
                    y=df_plot["anomaly_score"],
                    marker_color=["#FF4B4B" if p == "fail" else "#00C9A7" for p in df_plot["predicted"]],
                )
            )
            fig2.update_layout(**PLOTLY_LAYOUT, title="", height=280,
                               xaxis_title="Image #", yaxis_title="Anomaly Score",
                               margin=dict(l=30, r=10, t=10, b=30))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown(
                '<div style="height:200px;display:flex;align-items:center;justify-content:center;color:#4A6070;font-family:\'Share Tech Mono\',monospace;font-size:0.85rem;">Upload images to see score distribution</div>',
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown('<div class="section-header">Recent Activity</div>', unsafe_allow_html=True)
        if results:
            recent = results[-10:][::-1]
            for r in recent:
                badge = "🔴 FAIL" if r["is_defect"] else "🟢 PASS"
                fname = Path(r.get("name", "unknown")).name
                score_str = f"{r['score']:.4f}"
                st.markdown(
                    f"""
                    <div style='display:flex;justify-content:space-between;align-items:center;
                                padding:0.4rem 0.6rem;margin-bottom:0.3rem;
                                background:#0F2040;border-radius:6px;border:1px solid #1E3A5A;
                                font-family:"Share Tech Mono",monospace;font-size:0.75rem;'>
                        <span style='color:#D6E4F0;'>{fname[:20]}</span>
                        <span style='color:#5E7A96;'>{score_str}</span>
                        <span>{'<span style="color:#FF4B4B;">FAIL</span>' if r["is_defect"] else '<span style="color:#2ECC71;">PASS</span>'}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#4A6070;font-family:\'Share Tech Mono\',monospace;font-size:0.82rem;">No detections yet.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f'<div class="badge-info">Model: {"✓ Loaded" if model else "✗ Missing"}</div>'
            f'<div class="badge-info">Latent Dim: {cfg.get("model", {{}}).get("latent_dim", "—")}</div>',
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f'<div class="badge-info">Image Size: {cfg.get("model", {{}}).get("img_size", "—")}</div>'
            f'<div class="badge-info">Threshold mult: {cfg.get("evaluation", {{}}).get("threshold_multiplier", "—")}σ</div>',
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f'<div class="badge-info">Session detections: {total}</div>'
            f'<div class="badge-info">Threshold: {st.session_state["threshold"] or "not fitted"}</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — UPLOAD & DETECT
# ══════════════════════════════════════════════════════════════════════════════

def render_upload_detect():
    """Render the image upload and real-time defect detection page."""
    st.markdown('<div class="hero-title" style="font-size:1.8rem;">Upload & Detect</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">// SUBMIT IMAGES FOR ANOMALY SCORING</div>', unsafe_allow_html=True)
    st.markdown("")

    if model is None:
        st.error("⚠️ No trained model found. Please train the model first (Model Training page).")
        if st.button("→ Go to Model Training"):
            st.session_state["_nav"] = "🧠 Model Training"
        return

    # ── Threshold fitting ──────────────────────────────────────────────────────
    if evaluator is not None and st.session_state["threshold"] is None:
        normal_dir = cfg.get("paths", {}).get("normal_data", "data/normal/")
        normal_imgs = list(Path(normal_dir).glob("*.png")) + list(Path(normal_dir).glob("*.jpg"))
        if normal_imgs:
            with st.spinner("Fitting anomaly threshold from normal training data …"):
                import tensorflow as tf
                img_size = tuple(cfg["model"]["img_size"])

                def _parse(fp):
                    raw = tf.io.read_file(fp)
                    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
                    img = tf.image.resize(img, img_size)
                    img = tf.cast(img, tf.float32) / 255.0
                    img.set_shape([img_size[0], img_size[1], 3])
                    return img

                paths_str = [str(p) for p in normal_imgs[:100]]
                ds = (
                    tf.data.Dataset.from_tensor_slices(paths_str)
                    .map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(cfg["training"]["batch_size"])
                    .prefetch(tf.data.AUTOTUNE)
                )
                evaluator.fit_threshold(ds)
                st.session_state["threshold"] = evaluator.threshold
        else:
            # Use a reasonable default threshold
            evaluator.threshold = 0.01
            st.session_state["threshold"] = 0.01

    elif evaluator is not None and st.session_state["threshold"] is not None:
        evaluator.threshold = st.session_state["threshold"]

    col_upload, col_demo = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload inspection images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload one or more images for defect detection.",
        )
    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        use_demo = st.button("🧪 Use Demo Images", help="Load pre-generated normal and defect examples.")

    # Gather images to process
    images_to_process = []

    if use_demo:
        demo_normal = list(Path("data/normal").glob("*.png"))[:3]
        demo_defect = list(Path("data/defect").glob("*.png"))[:3]
        for p in demo_normal + demo_defect:
            images_to_process.append({"name": p.name, "bytes": p.read_bytes(), "type": "demo"})

    if uploaded:
        for f in uploaded:
            images_to_process.append({"name": f.name, "bytes": f.read(), "type": "upload"})

    if not images_to_process:
        st.markdown(
            """
            <div style='height:220px;display:flex;flex-direction:column;align-items:center;justify-content:center;
                        border:1px dashed #1E3A5A;border-radius:12px;margin-top:1rem;'>
                <div style='font-size:3rem;margin-bottom:1rem;opacity:0.3;'>🔬</div>
                <div style='color:#5E7A96;font-family:"Share Tech Mono",monospace;font-size:0.85rem;'>
                    Upload images or click "Use Demo Images" to begin inspection
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    all_results = []

    for item in images_to_process:
        with st.spinner(f"Inspecting {item['name']} …"):
            try:
                orig_u8, rec_u8, overlay_u8, score, is_defect = score_uploaded_image(
                    item["bytes"], evaluator, cfg
                )
            except Exception as e:
                st.error(f"Error processing {item['name']}: {e}")
                continue

        # Store in session history
        st.session_state["session_results"].append({
            "name": item["name"],
            "score": score,
            "is_defect": is_defect,
        })
        all_results.append({
            "name": item["name"],
            "score": score,
            "is_defect": is_defect,
            "orig": orig_u8,
            "rec": rec_u8,
            "overlay": overlay_u8,
        })

        # ── Per-image result card ──────────────────────────────────────────────
        st.markdown(f'<div class="section-header">{item["name"]}</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

        with col1:
            st.markdown('<div class="img-label">Original</div>', unsafe_allow_html=True)
            st.image(orig_u8, use_container_width=True)

        with col2:
            st.markdown('<div class="img-label">Reconstruction</div>', unsafe_allow_html=True)
            st.image(rec_u8, use_container_width=True)

        with col3:
            st.markdown('<div class="img-label">Error Heatmap</div>', unsafe_allow_html=True)
            st.image(overlay_u8, use_container_width=True)

        with col4:
            st.markdown('<div class="img-label">Result</div>', unsafe_allow_html=True)
            badge_cls = "badge-fail" if is_defect else "badge-pass"
            badge_txt = "⛔ DEFECT" if is_defect else "✅ PASS"
            threshold_val = st.session_state["threshold"] or 0.0
            confidence = float(1 / (1 + np.exp(-5 * (score - threshold_val) / (threshold_val + 1e-8))))
            conf_pct = int(confidence * 100) if is_defect else int((1 - confidence) * 100)
            st.markdown(
                f"""
                <div style='text-align:center;padding-top:1rem;'>
                    <div class="{badge_cls}">{badge_txt}</div>
                    <div style='margin-top:1rem;font-family:"Share Tech Mono",monospace;color:#5E7A96;font-size:0.8rem;'>
                        Anomaly Score<br>
                        <span style='font-size:1.5rem;color:#00C9A7;'>{score:.5f}</span>
                    </div>
                    <div style='margin-top:0.5rem;font-family:"Share Tech Mono",monospace;color:#5E7A96;font-size:0.75rem;'>
                        Confidence: <span style='color:#F39C12;'>{conf_pct}%</span>
                    </div>
                    <div style='margin-top:0.3rem;font-family:"Share Tech Mono",monospace;color:#4A6070;font-size:0.68rem;'>
                        Threshold: {threshold_val:.5f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

    st.toast(f"✅ Processed {len(all_results)} image(s).", icon="🔬")

    # ── Batch export ───────────────────────────────────────────────────────────
    if all_results:
        st.markdown("---")
        export_df = pd.DataFrame([
            {
                "filename": r["name"],
                "anomaly_score": r["score"],
                "result": "FAIL" if r["is_defect"] else "PASS",
                "threshold": st.session_state["threshold"],
            }
            for r in all_results
        ])
        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download Results CSV",
            data=csv_bytes,
            file_name=f"visualsentry_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def render_training():
    """Render the model training configuration and execution page."""
    st.markdown('<div class="hero-title" style="font-size:1.8rem;">Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">// CONFIGURE AND LAUNCH THE TRAINING PIPELINE</div>', unsafe_allow_html=True)
    st.markdown("")

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    with st.expander("⚙️ Hyperparameter Configuration", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        epochs = col1.number_input("Epochs", min_value=1, max_value=200, value=int(train_cfg.get("epochs", 20)))
        batch_size = col2.number_input("Batch Size", min_value=4, max_value=128, step=4, value=int(train_cfg.get("batch_size", 32)))
        latent_dim = col3.number_input("Latent Dim", min_value=8, max_value=512, step=8, value=int(model_cfg.get("latent_dim", 64)))
        learning_rate = col4.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=float(train_cfg.get("learning_rate", 0.001)), format="%.4f")

    gen_data = st.checkbox("Generate demo data before training", value=True)

    if st.button("🚀 Start Training", type="primary"):
        if gen_data:
            with st.spinner("Generating synthetic demo dataset …"):
                try:
                    from src.data_loader import generate_demo_data
                    generate_demo_data("config.yaml")
                    st.toast("Demo data generated.", icon="✅")
                except Exception as e:
                    st.error(f"Data generation failed: {e}")
                    return

        progress_bar = st.progress(0, text="Initialising …")
        loss_placeholder = st.empty()
        epoch_log = []

        def epoch_cb(epoch: int, logs: dict):
            """Update Streamlit progress bar and loss display after each epoch."""
            pct = min(int((epoch + 1) / epochs * 100), 100)
            train_l = logs.get("loss", 0)
            val_l = logs.get("val_loss", 0)
            epoch_log.append({"epoch": epoch + 1, "loss": train_l, "val_loss": val_l})
            progress_bar.progress(pct, text=f"Epoch {epoch+1}/{epochs} — train={train_l:.5f} val={val_l:.5f}")

            if len(epoch_log) >= 2:
                df_live = pd.DataFrame(epoch_log)
                from src.utils import plot_training_loss
                fig = plot_training_loss(df_live, title="Live Training Loss")
                fig.update_layout(height=260, margin=dict(l=30, r=10, t=20, b=30))
                loss_placeholder.plotly_chart(fig, use_container_width=True)

        try:
            from src.trainer import Trainer
            import yaml

            # Patch config in-memory with GUI values
            runtime_cfg = get_config()
            runtime_cfg["training"]["epochs"] = epochs
            runtime_cfg["training"]["batch_size"] = batch_size
            runtime_cfg["training"]["learning_rate"] = learning_rate
            runtime_cfg["model"]["latent_dim"] = latent_dim

            # Write patched config to a temp file
            import tempfile, yaml as _yaml
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=".") as tmp:
                _yaml.dump(runtime_cfg, tmp)
                tmp_path = tmp.name

            trainer = Trainer(config_path=tmp_path)
            trainer.build_model()
            history = trainer.train(progress_callback=epoch_cb)
            Path(tmp_path).unlink(missing_ok=True)

            progress_bar.progress(100, text="Training complete ✅")
            st.toast("Model trained and saved!", icon="🧠")

            # Reload model cache
            st.cache_resource.clear()

            # Show final loss curve
            log_df = trainer.get_training_log()
            if log_df is not None:
                from src.utils import plot_training_loss
                fig = plot_training_loss(log_df, title="Final Training History")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Training failed: {e}")
            return

    st.markdown("---")
    st.markdown('<div class="section-header">Model Architecture</div>', unsafe_allow_html=True)

    model_summary = f"""
ConvAutoencoder Summary
=======================
Input         : ({model_cfg.get('img_size', [128,128])[0]}, {model_cfg.get('img_size', [128,128])[1]}, 3)
Encoder
  Conv2D-32   : (128, 128, 32) → MaxPool → (64, 64, 32)
  Conv2D-64   : (64, 64, 64)   → MaxPool → (32, 32, 64)
  Conv2D-128  : (32, 32, 128)  → MaxPool → (16, 16, 128)
  Flatten     : 32768
  Dense       : {latent_dim}                   ← Bottleneck
Decoder
  Dense       : 32768
  Reshape     : (16, 16, 128)
  ConvT-128   : (32, 32, 128)  (stride=2)
  ConvT-64    : (64, 64, 64)   (stride=2)
  ConvT-32    : (128, 128, 32) (stride=2)
  Conv2D-3    : (128, 128, 3)  sigmoid  ← Reconstruction
"""
    st.code(model_summary, language=None)

    # Download model button
    mp = Path(model_path)
    if mp.exists():
        with open(mp, "rb") as fh:
            st.download_button(
                "⬇ Download Trained Model (.h5)",
                data=fh.read(),
                file_name=mp.name,
                mime="application/octet-stream",
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def render_analytics():
    """Render the analytics and evaluation metrics dashboard."""
    st.markdown('<div class="hero-title" style="font-size:1.8rem;">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">// EVALUATION METRICS & PERFORMANCE ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown("")

    eval_csv = cfg.get("paths", {}).get("evaluation_results", "reports/evaluation_results.csv")

    if not Path(eval_csv).exists():
        st.warning("No evaluation results found. Run `python evaluate.py` or process images on the Upload page first.")
        st.markdown(
            """
            <div style='font-family:"Share Tech Mono",monospace;color:#5E7A96;font-size:0.82rem;margin-top:1rem;'>
            Quick start:<br>
            <code>make data && make train && make evaluate</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    results_df = pd.read_csv(eval_csv)
    scores = results_df["anomaly_score"].values
    labels = [1 if lbl == "defect" else 0 for lbl in results_df["label"]]
    predicted_bin = [1 if p == "fail" else 0 for p in results_df["predicted"]]

    if model is not None and evaluator is not None:
        if st.session_state["threshold"] is None:
            evaluator.threshold = float(scores[np.array(labels) == 0].mean() + 2 * scores[np.array(labels) == 0].std())
            st.session_state["threshold"] = evaluator.threshold
        else:
            evaluator.threshold = st.session_state["threshold"]

    from src.utils import (
        plot_score_distribution, plot_roc_curve, plot_pr_curve,
        plot_confusion_matrix, compute_confusion_values
    )

    # ── Top metrics ───────────────────────────────────────────────────────────
    from src.evaluator import AnomalyEvaluator
    tmp_eval = AnomalyEvaluator.__new__(AnomalyEvaluator)
    tmp_eval.threshold = st.session_state.get("threshold", 0.01)
    metrics = tmp_eval.compute_metrics(labels, predicted_bin, scores) if model else {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc_roc": 0.0
    }
    from src.evaluator import AnomalyEvaluator as _AE
    _AE.__init__  # suppress linter

    from src.utils import format_metric_card
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(format_metric_card("Precision", f"{metrics['precision']:.3f}"), unsafe_allow_html=True)
    m2.markdown(format_metric_card("Recall", f"{metrics['recall']:.3f}"), unsafe_allow_html=True)
    m3.markdown(format_metric_card("F1-Score", f"{metrics['f1']:.3f}"), unsafe_allow_html=True)
    m4.markdown(format_metric_card("AUC-ROC", f"{metrics['auc_roc']:.3f}"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Score distribution ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig_dist = plot_score_distribution(
            scores, labels, threshold=st.session_state.get("threshold"),
            title="Anomaly Score Distribution"
        )
        fig_dist.update_layout(height=320)
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Confusion matrix ───────────────────────────────────────────────────────
    with col2:
        cv = compute_confusion_values(labels, predicted_bin)
        fig_cm = plot_confusion_matrix(cv["tp"], cv["tn"], cv["fp"], cv["fn"], title="Confusion Matrix")
        fig_cm.update_layout(height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC + PR curves ────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    if len(np.unique(labels)) > 1:
        from src.evaluator import AnomalyEvaluator as _AE2
        _tmp2 = _AE2.__new__(_AE2)
        _tmp2.threshold = st.session_state.get("threshold", 0.01)

        try:
            from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
            fpr, tpr, _ = roc_curve(labels, scores)
            auc_val = roc_auc_score(labels, scores)
            with col3:
                fig_roc = plot_roc_curve(fpr, tpr, auc_val, title="ROC Curve")
                fig_roc.update_layout(height=320)
                st.plotly_chart(fig_roc, use_container_width=True)

            prec, rec, _ = precision_recall_curve(labels, scores)
            with col4:
                fig_pr = plot_pr_curve(prec, rec, title="Precision–Recall Curve")
                fig_pr.update_layout(height=320)
                st.plotly_chart(fig_pr, use_container_width=True)
        except Exception:
            pass

    # ── Highest anomaly images grid ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Highest Anomaly Score Samples</div>', unsafe_allow_html=True)

    top_idx = results_df.nlargest(9, "anomaly_score").index.tolist()
    cols_grid = st.columns(3)
    for i, idx in enumerate(top_idx):
        row = results_df.loc[idx]
        img_path = row["image_path"]
        if Path(img_path).exists():
            pil = Image.open(img_path).convert("RGB")
            with cols_grid[i % 3]:
                st.image(pil, caption=f"Score: {row['anomaly_score']:.4f} | {row['predicted'].upper()}", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def render_about():
    """Render the about / project documentation page."""
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-title">VisualSentry</div>
            <div class="hero-subtitle">// AI-POWERED VISUAL DEFECT DETECTION FOR MANUFACTURING QC</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        VisualSentry is a production-grade, unsupervised anomaly detection system built on a Convolutional
        Autoencoder (CAE) trained exclusively on normal surface images. At inference time, defective images
        produce anomalously high reconstruction error — which is thresholded to issue pass/fail decisions
        in real time. No defect labels are required during training.
        """,
    )

    st.markdown("---")
    st.markdown('<div class="section-header">Architecture Pipeline</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ```
        Raw Image  ──►  Preprocess  ──►  Encoder  ──►  Latent z  ──►  Decoder
        (128×128)        (resize,           ↓           (dim=64)        ↓
                         normalise)     Conv2D×3                   Conv2DT×3
                                        MaxPool×3                       ↓
                                            ↓                    Reconstruction
                                         Dense                        ↓
                                            ↓                   MSE Error Map
                                         Latent                       ↓
                                                              Anomaly Score
                                                                     ↓
                                                             μ + 2σ  Threshold
                                                                     ↓
                                                            ✅ PASS / ⛔ FAIL
        ```
        """
    )

    st.markdown("---")
    st.markdown('<div class="section-header">Key Features</div>', unsafe_allow_html=True)

    features = [
        ("🏭", "Unsupervised Detection", "No defect labels needed during training — learns normality only."),
        ("⚡", "Real-Time Scoring", "Sub-second inference via optimised Keras model on CPU."),
        ("📊", "Adaptive Threshold", "Statistical threshold computed as μ + k·σ over normal val images."),
        ("🌡️", "Reconstruction Heatmaps", "Per-pixel error maps overlaid on original images for interpretability."),
        ("⚙️", "Full MLOps Pipeline", "YAML config, CSV logging, early stopping, model checkpointing."),
        ("🎛️", "Interactive GUI", "Professional Streamlit dashboard with Plotly analytics."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="metric-card" style='text-align:left;margin-bottom:1rem;'>
                    <div style='font-size:1.5rem;margin-bottom:0.4rem;'>{icon}</div>
                    <div style='font-family:"Share Tech Mono",monospace;color:#00C9A7;font-size:0.85rem;margin-bottom:0.3rem;'>{title}</div>
                    <div style='font-size:0.8rem;color:#5E7A96;'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="section-header">Tech Stack</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style='margin-top:0.5rem;'>
            <span class="tech-pill pill-py">Python 3.10</span>
            <span class="tech-pill pill-tf">TensorFlow 2.13</span>
            <span class="tech-pill pill-st">Streamlit 1.28</span>
            <span class="tech-pill pill-sk">scikit-learn</span>
            <span class="tech-pill pill-np">NumPy</span>
            <span class="tech-pill pill-py">Plotly</span>
            <span class="tech-pill pill-tf">OpenCV</span>
            <span class="tech-pill pill-sk">Pandas</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="section-header">Links</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(
        '<a href="https://github.com/your-username/VisualSentry" target="_blank" style="color:#00C9A7;font-family:\'Share Tech Mono\',monospace;">🐙 GitHub Repository</a>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        '<a href="https://linkedin.com/in/your-profile" target="_blank" style="color:#00A8E8;font-family:\'Share Tech Mono\',monospace;">💼 LinkedIn Profile</a>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(
        '<div style="font-family:\'Share Tech Mono\',monospace;color:#2A3D50;font-size:0.72rem;text-align:center;margin-top:2rem;">VisualSentry © 2024 — MIT License — Detect what the human eye misses.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Dashboard":
    render_dashboard()
elif page == "🔍 Upload & Detect":
    render_upload_detect()
elif page == "🧠 Model Training":
    render_training()
elif page == "📈 Analytics":
    render_analytics()
elif page == "ℹ️ About":
    render_about()
