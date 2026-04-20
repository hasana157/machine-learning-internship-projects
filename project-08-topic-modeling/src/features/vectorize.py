"""
vectorize.py — Feature engineering: TF-IDF and Count Vectorization.

Exposes a single ``build_vectorizer`` factory and helper functions to
fit, transform, and persist vectorizers.  Kept entirely separate from
model training to honour single-responsibility.
"""

import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import spmatrix

from src.utils import get_logger, path_for

logger = get_logger(__name__)

# Type alias
Vectorizer = TfidfVectorizer | CountVectorizer


# ── Factory ───────────────────────────────────────────────────────────────────

def build_vectorizer(
    kind: str = "tfidf",
    *,
    max_features: int = 5000,
    max_df: float = 0.95,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (1, 1),
    **kwargs,
) -> Vectorizer:
    """
    Instantiate a TF-IDF or Count vectorizer with sensible defaults.

    Parameters
    ----------
    kind : str
        ``"tfidf"`` (default) or ``"count"``.
    max_features : int
        Vocabulary size cap.
    max_df : float
        Ignore terms appearing in more than this fraction of documents.
    min_df : int
        Ignore terms appearing in fewer than this many documents.
    ngram_range : tuple
        The lower and upper boundary of the n-gram range.
    **kwargs
        Extra keyword arguments forwarded to the underlying sklearn class.

    Returns
    -------
    TfidfVectorizer | CountVectorizer

    Raises
    ------
    ValueError
        If ``kind`` is not recognised.
    """
    shared = dict(
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",   # letters only, ≥3 chars
        **kwargs,
    )

    if kind == "tfidf":
        vec = TfidfVectorizer(
            sublinear_tf=True,          # log(1 + tf) scaling
            norm="l2",
            **shared,
        )
        logger.info("Built TF-IDF vectorizer (max_features=%d).", max_features)
    elif kind == "count":
        vec = CountVectorizer(**shared)
        logger.info("Built Count vectorizer (max_features=%d).", max_features)
    else:
        raise ValueError(f"Unknown vectorizer kind '{kind}'. Choose 'tfidf' or 'count'.")

    return vec


# ── Fit / transform helpers ───────────────────────────────────────────────────

def fit_vectorizer(vec: Vectorizer, corpus: list[str]) -> Vectorizer:
    """
    Fit a vectorizer on the supplied corpus in place.

    Parameters
    ----------
    vec : Vectorizer
        An un-fitted sklearn vectorizer.
    corpus : list of str
        Preprocessed documents.

    Returns
    -------
    Vectorizer
        The same object, now fitted.
    """
    logger.info("Fitting vectorizer on %d documents …", len(corpus))
    vec.fit(corpus)
    logger.info("Vocabulary size: %d", len(vec.vocabulary_))
    return vec


def transform_corpus(vec: Vectorizer, corpus: list[str]) -> spmatrix:
    """
    Transform a corpus into a document-term matrix.

    Parameters
    ----------
    vec : Vectorizer
        A fitted vectorizer.
    corpus : list of str
        Documents to transform (may differ from training set).

    Returns
    -------
    scipy.sparse matrix  (n_docs × n_features)
    """
    dtm = vec.transform(corpus)
    logger.info(
        "Document-term matrix shape: %s, nnz=%d",
        dtm.shape,
        dtm.nnz,
    )
    return dtm


def fit_transform_corpus(
    vec: Vectorizer, corpus: list[str]
) -> tuple[Vectorizer, spmatrix]:
    """
    Combined fit + transform in one call.

    Returns
    -------
    (fitted_vectorizer, document_term_matrix)
    """
    fit_vectorizer(vec, corpus)
    dtm = transform_corpus(vec, corpus)
    return vec, dtm


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_vectorizer(vec: Vectorizer, filename: str = "vectorizer.joblib") -> Path:
    """
    Persist a fitted vectorizer to ``models/``.

    Parameters
    ----------
    vec : Vectorizer
    filename : str

    Returns
    -------
    Path
    """
    out = path_for("models", filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, out)
    logger.info("Vectorizer saved → %s", out)
    return out


def load_vectorizer(filename: str = "vectorizer.joblib") -> Vectorizer:
    """
    Load a previously saved vectorizer.

    Parameters
    ----------
    filename : str

    Returns
    -------
    Vectorizer

    Raises
    ------
    FileNotFoundError
    """
    file_path = path_for("models", filename)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Vectorizer not found at {file_path}. "
            "Run the training pipeline first."
        )
    vec = joblib.load(file_path)
    logger.info("Vectorizer loaded ← %s", file_path)
    return vec


def get_feature_names(vec: Vectorizer) -> list[str]:
    """Return vocabulary as a plain list (sklearn ≥ 1.0 compatible)."""
    return vec.get_feature_names_out().tolist()
