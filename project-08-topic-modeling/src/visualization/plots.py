"""
plots.py — Reusable, publication-quality visualisation functions.

All public functions accept a ``save_path`` argument; when provided,
the figure is written to disk.  Figures are also returned so callers
can display them interactively (e.g. in a Jupyter notebook).

Design choices
--------------
- Consistent colour palette and font settings applied at module import.
- Every function is independent — no shared mutable state.
- Matplotlib used as the primary backend; WordCloud for word clouds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server / CI use
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from collections import Counter

from src.utils import get_logger, path_for

logger = get_logger(__name__)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "#f9f9f9",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
    }
)

_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
    "#3B1F2B", "#44BBA4", "#E94F37", "#393E41",
    "#F5A623", "#7B2FBE",
]


def _save(fig: plt.Figure, save_path: Path | str | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Figure saved → %s", save_path)


# ── EDA plots ─────────────────────────────────────────────────────────────────

def plot_doc_length_distribution(
    texts: list[str],
    save_path: Path | str | None = None,
    title: str = "Document Length Distribution (Word Count)",
) -> plt.Figure:
    """
    Histogram of per-document word counts.

    Parameters
    ----------
    texts : list of str
    save_path : path-like, optional
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    lengths = [len(t.split()) for t in texts]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(lengths, bins=60, color=_PALETTE[0], edgecolor="white", alpha=0.85)
    ax.axvline(np.median(lengths), color=_PALETTE[2], lw=2,
               linestyle="--", label=f"Median = {np.median(lengths):.0f}")
    ax.set_xlabel("Words per document")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_category_distribution(
    category_names: list[str],
    save_path: Path | str | None = None,
    title: str = "Document Count per Newsgroup Category",
) -> plt.Figure:
    """
    Horizontal bar chart of category frequencies.

    Parameters
    ----------
    category_names : list of str
    save_path : path-like, optional
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    counts = Counter(category_names)
    cats, vals = zip(*sorted(counts.items(), key=lambda x: x[1]))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(cats))]
    bars = ax.barh(cats, vals, color=colors, edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=8)
    ax.set_xlabel("Number of documents")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_top_words_global(
    texts: list[str],
    n: int = 30,
    save_path: Path | str | None = None,
    title: str = "Top Words in Corpus (after preprocessing)",
) -> plt.Figure:
    """
    Bar chart of the most frequent words across the whole corpus.

    Parameters
    ----------
    texts : list of str
        Pre-processed documents (space-separated tokens).
    n : int
        Number of top words to display.
    save_path : path-like, optional
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    all_tokens: list[str] = []
    for doc in texts:
        all_tokens.extend(doc.split())

    counts = Counter(all_tokens).most_common(n)
    words, freqs = zip(*counts)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(words, freqs, color=_PALETTE[1], edgecolor="white")
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Topic model plots ─────────────────────────────────────────────────────────

def plot_topic_top_words(
    topics: list[dict],
    n_cols: int = 5,
    n_top: int = 10,
    save_path: Path | str | None = None,
    title: str = "Top Words per Topic",
) -> plt.Figure:
    """
    Grid of horizontal bar charts — one subplot per topic.

    Parameters
    ----------
    topics : list of dict
        As returned by :func:`src.models.train.get_top_words`.
    n_cols : int
        Columns in the subplot grid.
    n_top : int
        Words to display per topic (≤ len of words list).
    save_path : path-like, optional
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_topics = len(topics)
    n_rows = (n_topics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
    axes = np.array(axes).flatten()

    for ax, topic in zip(axes, topics):
        words = topic["words"][:n_top]
        weights = topic["weights"][:n_top]
        # Reverse so highest-weight word is at top
        ax.barh(words[::-1], weights[::-1],
                color=_PALETTE[topic["topic_id"] % len(_PALETTE)])
        ax.set_title(topic["label"], fontsize=9, fontweight="bold")
        ax.set_xlabel("Weight", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Hide unused axes
    for ax in axes[n_topics:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_topic_distribution_heatmap(
    topic_matrix: np.ndarray,
    save_path: Path | str | None = None,
    title: str = "Topic Distribution Heatmap (sample)",
    n_sample: int = 100,
) -> plt.Figure:
    """
    Heatmap of topic weights for a random document sample.

    Parameters
    ----------
    topic_matrix : np.ndarray  (n_docs × n_topics)
    save_path : path-like, optional
    title : str
    n_sample : int
        Number of documents to sample (for readability).

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng = np.random.default_rng(42)
    idx = rng.choice(len(topic_matrix), size=min(n_sample, len(topic_matrix)),
                     replace=False)
    sample = topic_matrix[idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(sample.T, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Document (sample)")
    ax.set_ylabel("Topic")
    ax.set_yticks(range(sample.shape[1]))
    ax.set_yticklabels([f"T{i}" for i in range(sample.shape[1])], fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Weight")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_dominant_topic_counts(
    dominant_topics: list[int],
    n_topics: int,
    save_path: Path | str | None = None,
    title: str = "Document Count per Dominant Topic",
) -> plt.Figure:
    """
    Bar chart of how many documents have each topic as dominant.

    Parameters
    ----------
    dominant_topics : list of int
    n_topics : int
    save_path : path-like, optional
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    counts = Counter(dominant_topics)
    x = list(range(n_topics))
    y = [counts.get(i, 0) for i in x]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        [f"T{i}" for i in x], y,
        color=[_PALETTE[i % len(_PALETTE)] for i in x],
        edgecolor="white",
    )
    ax.set_xlabel("Topic")
    ax.set_ylabel("Document count")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_sweep_results(
    sweep_df: pd.DataFrame,
    save_path: Path | str | None = None,
    title: str = "Model Quality vs. Number of Topics",
) -> plt.Figure:
    """
    Line plot of reconstruction error (or perplexity) and diversity
    across different n_topic values.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of :func:`src.models.evaluate.sweep_n_topics`.
    save_path : path-like, optional
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    error_col = (
        "reconstruction_error" if "reconstruction_error" in sweep_df.columns
        else "perplexity"
    )
    error_label = "Reconstruction Error" if error_col == "reconstruction_error" else "Perplexity"

    ax1.plot(sweep_df["n_topics"], sweep_df[error_col],
             marker="o", color=_PALETTE[0], lw=2)
    ax1.set_xlabel("Number of Topics")
    ax1.set_ylabel(error_label)
    ax1.set_title(f"{error_label} vs n_topics")

    ax2.plot(sweep_df["n_topics"], sweep_df["diversity"],
             marker="s", color=_PALETTE[1], lw=2)
    ax2.set_xlabel("Number of Topics")
    ax2.set_ylabel("Topic Diversity")
    ax2.set_title("Topic Diversity vs n_topics")

    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_wordcloud(
    words: list[str],
    weights: list[float] | None = None,
    save_path: Path | str | None = None,
    title: str = "Word Cloud",
    max_words: int = 80,
) -> plt.Figure:
    """
    Generate a word cloud from a list of words (and optional weights).

    Parameters
    ----------
    words : list of str
    weights : list of float, optional
        Relative frequencies; uniform if not provided.
    save_path : path-like, optional
    title : str
    max_words : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud package not installed; skipping word cloud.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Install wordcloud package", ha="center", va="center")
        return fig

    freq_dict = (
        dict(zip(words, weights)) if weights is not None
        else {w: 1 for w in words}
    )

    wc = WordCloud(
        width=900, height=450,
        background_color="white",
        max_words=max_words,
        colormap="tab20",
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Quick helper for batch EDA ─────────────────────────────────────────────────

def save_all_eda_plots(df: "pd.DataFrame") -> None:  # noqa: F821
    """
    Generate and save the standard EDA figure set.

    Parameters
    ----------
    df : pd.DataFrame
        Processed corpus DataFrame with columns
        ``[raw_text, clean_text, category_name]``.
    """
    fig_dir = path_for("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_doc_length_distribution(
        df["raw_text"].tolist(),
        save_path=fig_dir / "raw_doc_lengths.png",
        title="Raw Document Length Distribution",
    )
    plot_doc_length_distribution(
        df["clean_text"].tolist(),
        save_path=fig_dir / "clean_doc_lengths.png",
        title="Cleaned Document Length Distribution",
    )
    plot_category_distribution(
        df["category_name"].tolist(),
        save_path=fig_dir / "category_distribution.png",
    )
    plot_top_words_global(
        df["clean_text"].tolist(),
        n=30,
        save_path=fig_dir / "top_words_global.png",
    )
    logger.info("All EDA plots saved to %s", fig_dir)
