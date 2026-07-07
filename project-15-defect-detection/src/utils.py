"""
utils.py
========
Shared utility functions for VisualSentry: image I/O helpers, Plotly chart
builders, and miscellaneous pipeline utilities used across the codebase.
"""

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import yaml

logger = logging.getLogger(__name__)

# ── Brand colour palette ───────────────────────────────────────────────────────
PALETTE = {
    "bg": "#0D1B2A",
    "surface": "#112233",
    "accent": "#00C9A7",
    "danger": "#FF4B4B",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "text": "#E8EDF2",
    "muted": "#6B7B8D",
    "grid": "#1A2B3C",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["surface"],
    font=dict(color=PALETTE["text"], family="'Courier New', monospace"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"]),
    yaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"]),
)


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Image utilities ────────────────────────────────────────────────────────────

def load_pil_image(path: str, img_size: Tuple[int, int]) -> Image.Image:
    """Load and resize a PIL Image from disk.

    Args:
        path: File system path to the image.
        img_size: Target (height, width).

    Returns:
        RGB PIL Image resized to img_size.
    """
    return Image.open(path).convert("RGB").resize((img_size[1], img_size[0]))


def pil_to_numpy(pil_img: Image.Image, normalise: bool = True) -> np.ndarray:
    """Convert a PIL Image to a float32 numpy array.

    Args:
        pil_img: Input PIL Image.
        normalise: If True, scale pixel values to [0, 1].

    Returns:
        Float32 ndarray of shape (H, W, 3).
    """
    arr = np.array(pil_img, dtype=np.float32)
    if normalise:
        arr /= 255.0
    return arr


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a float32 numpy array to a PIL Image.

    Args:
        arr: Float32 ndarray of shape (H, W, 3) in [0, 1].

    Returns:
        RGB PIL Image.
    """
    uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(uint8)


def bytes_to_numpy(file_bytes: bytes, img_size: Tuple[int, int]) -> np.ndarray:
    """Decode uploaded image bytes into a normalised float32 ndarray.

    Args:
        file_bytes: Raw bytes of an image file.
        img_size: Target (height, width) for resizing.

    Returns:
        Float32 ndarray of shape (H, W, 3) in [0, 1].
    """
    pil = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((img_size[1], img_size[0]))
    return pil_to_numpy(pil, normalise=True)


def list_images(directory: str) -> List[str]:
    """List all image file paths in a directory (non-recursive).

    Args:
        directory: Path to the directory to scan.

    Returns:
        Sorted list of absolute path strings for jpg/jpeg/png files.
    """
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(str(p) for p in Path(directory).glob("*") if p.suffix.lower() in exts)


# ── Plotly chart helpers ───────────────────────────────────────────────────────

def plot_score_distribution(
    scores: np.ndarray,
    labels: List[int],
    threshold: Optional[float] = None,
    title: str = "Anomaly Score Distribution",
) -> go.Figure:
    """Plot a histogram of anomaly scores coloured by class.

    Args:
        scores: Array of per-image anomaly scores.
        labels: Binary labels (0=normal, 1=defect).
        threshold: Decision threshold to draw as a vertical line.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    df = pd.DataFrame({"score": scores, "label": ["Defect" if l else "Normal" for l in labels]})

    fig = px.histogram(
        df,
        x="score",
        color="label",
        barmode="overlay",
        nbins=40,
        color_discrete_map={"Normal": PALETTE["accent"], "Defect": PALETTE["danger"]},
        title=title,
        opacity=0.75,
    )
    fig.update_layout(**PLOTLY_LAYOUT)

    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=PALETTE["warning"],
            annotation_text=f"Threshold {threshold:.4f}",
            annotation_font_color=PALETTE["warning"],
        )

    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    title: str = "ROC Curve",
) -> go.Figure:
    """Plot a styled ROC curve with AUC annotation.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc: Area under the ROC curve.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC = {auc:.3f}",
            line=dict(color=PALETTE["accent"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,201,167,0.10)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color=PALETTE["muted"], dash="dash", width=1.5),
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    title: str = "Precision–Recall Curve",
) -> go.Figure:
    """Plot a styled Precision-Recall curve.

    Args:
        precision: Precision values at each threshold.
        recall: Recall values at each threshold.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="PR Curve",
            line=dict(color=PALETTE["warning"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(243,156,18,0.10)",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    return fig


def plot_confusion_matrix(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Plot a styled confusion matrix heatmap.

    Args:
        tp: True positives.
        tn: True negatives.
        fp: False positives.
        fn: False negatives.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    z = [[tn, fp], [fn, tp]]
    annotations = [
        [f"TN<br>{tn}", f"FP<br>{fp}"],
        [f"FN<br>{fn}", f"TP<br>{tp}"],
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predicted Normal", "Predicted Defect"],
            y=["Actual Normal", "Actual Defect"],
            colorscale=[[0, "#112233"], [0.5, "#00C9A7"], [1.0, "#FF4B4B"]],
            showscale=False,
            text=annotations,
            texttemplate="%{text}",
            textfont=dict(color="white", size=16),
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, title=title)
    return fig


def plot_training_loss(log_df: pd.DataFrame, title: str = "Training History") -> go.Figure:
    """Plot training and validation loss from a CSV log DataFrame.

    Args:
        log_df: DataFrame with 'epoch', 'loss', and optionally 'val_loss' columns.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=log_df["epoch"],
            y=log_df["loss"],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color=PALETTE["accent"], width=2),
            marker=dict(size=5),
        )
    )
    if "val_loss" in log_df.columns:
        fig.add_trace(
            go.Scatter(
                x=log_df["epoch"],
                y=log_df["val_loss"],
                mode="lines+markers",
                name="Val Loss",
                line=dict(color=PALETTE["danger"], width=2, dash="dash"),
                marker=dict(size=5),
            )
        )
    fig.update_layout(**PLOTLY_LAYOUT, title=title, xaxis_title="Epoch", yaxis_title="MSE Loss")
    return fig


def plot_score_bar(results_df: pd.DataFrame, title: str = "Per-Image Anomaly Scores") -> go.Figure:
    """Bar chart of anomaly scores per image coloured by pass/fail.

    Args:
        results_df: DataFrame with 'anomaly_score' and 'predicted' columns.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    color_map = {"pass": PALETTE["accent"], "fail": PALETTE["danger"]}
    colors = [color_map.get(p, PALETTE["muted"]) for p in results_df["predicted"]]

    fig = go.Figure(
        go.Bar(
            x=list(range(len(results_df))),
            y=results_df["anomaly_score"],
            marker_color=colors,
            name="Anomaly Score",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, title=title, xaxis_title="Image Index", yaxis_title="Score")
    return fig


# ── Metrics helpers ────────────────────────────────────────────────────────────

def compute_confusion_values(
    labels: List[int], predicted: List[int]
) -> Dict[str, int]:
    """Compute TP, TN, FP, FN from binary label lists.

    Args:
        labels: Ground-truth binary labels (0=normal, 1=defect).
        predicted: Predicted binary labels.

    Returns:
        Dictionary with keys 'tp', 'tn', 'fp', 'fn'.
    """
    labels_arr = np.array(labels)
    pred_arr = np.array(predicted)
    return {
        "tp": int(np.sum((labels_arr == 1) & (pred_arr == 1))),
        "tn": int(np.sum((labels_arr == 0) & (pred_arr == 0))),
        "fp": int(np.sum((labels_arr == 0) & (pred_arr == 1))),
        "fn": int(np.sum((labels_arr == 1) & (pred_arr == 0))),
    }


def format_metric_card(label: str, value: str, delta: Optional[str] = None) -> str:
    """Generate styled HTML for a metric card displayed in Streamlit.

    Args:
        label: Metric label text.
        value: Primary metric value.
        delta: Optional delta / secondary string.

    Returns:
        HTML string ready for st.markdown(..., unsafe_allow_html=True).
    """
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
