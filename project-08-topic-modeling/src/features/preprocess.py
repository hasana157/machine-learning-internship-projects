"""
preprocess.py — Text cleaning and normalisation pipeline.

Provides both a functional API (``clean_text``, ``clean_corpus``) and a
scikit-learn-compatible ``TextPreprocessor`` transformer so the preprocessing
step can be embedded inside sklearn ``Pipeline`` objects.
"""

import re
import string
from typing import Callable

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils import get_logger

logger = get_logger(__name__)

# ── Download NLTK data (idempotent) ──────────────────────────────────────────
_NLTK_RESOURCES = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]

def _ensure_nltk() -> None:
    for resource in _NLTK_RESOURCES:
        try:
            if resource in ("punkt", "punkt_tab"):
                nltk.data.find(f"tokenizers/{resource}")
            elif resource in ("stopwords",):
                nltk.data.find(f"corpora/{resource}")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)

_ensure_nltk()

# ── Module-level singletons ───────────────────────────────────────────────────
_STOPWORDS: frozenset[str] = frozenset(stopwords.words("english"))
_LEMMATIZER: WordNetLemmatizer = WordNetLemmatizer()

# Extra domain-specific noise words common in 20 Newsgroups
_EXTRA_STOPS: frozenset[str] = frozenset(
    {
        "subject", "re", "edu", "use", "com", "would", "could", "also",
        "get", "like", "one", "may", "make", "know", "think", "say",
        "go", "way", "well", "even", "still", "see", "much", "many",
        "used", "good", "us", "article", "writes", "wrote", "said",
    }
)

ALL_STOPS: frozenset[str] = _STOPWORDS | _EXTRA_STOPS


# ── Core cleaning functions ───────────────────────────────────────────────────

def remove_email_artifacts(text: str) -> str:
    """Strip e-mail addresses, URLs, and common header patterns."""
    # Remove e-mail addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove lines that look like e-mail headers (From:, Subject:, etc.)
    text = re.sub(r"^(from|subject|organization|lines|nntp-posting-host):.*$",
                  " ", text, flags=re.MULTILINE | re.IGNORECASE)
    return text


def remove_punctuation(text: str) -> str:
    """Replace punctuation and digits with a space."""
    translator = str.maketrans(string.punctuation + string.digits,
                               " " * (len(string.punctuation) + len(string.digits)))
    return text.translate(translator)


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    min_token_len: int = 3,
) -> str:
    """
    Full single-document cleaning pipeline.

    Parameters
    ----------
    text : str
        Raw document string.
    lowercase : bool
        Convert to lowercase (default True).
    remove_stopwords : bool
        Remove English stop-words (default True).
    lemmatize : bool
        Apply WordNet lemmatisation (default True).
    min_token_len : int
        Discard tokens shorter than this (default 3).

    Returns
    -------
    str
        Space-joined cleaned token string.
    """
    if not isinstance(text, str):
        return ""

    text = remove_email_artifacts(text)
    text = remove_punctuation(text)

    if lowercase:
        text = text.lower()

    tokens: list[str] = word_tokenize(text)

    # Length filter
    tokens = [t for t in tokens if len(t) >= min_token_len]

    # Stop-word removal
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ALL_STOPS]

    # Lemmatisation
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    # Second pass stop-word check after lemmatisation
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ALL_STOPS]

    return " ".join(tokens)


def clean_corpus(
    documents: list[str],
    **kwargs,
) -> list[str]:
    """
    Apply :func:`clean_text` to every document in a corpus.

    Parameters
    ----------
    documents : list of str
        Raw documents.
    **kwargs
        Passed through to :func:`clean_text`.

    Returns
    -------
    list of str
    """
    cleaned = []
    for i, doc in enumerate(documents):
        cleaned.append(clean_text(doc, **kwargs))
        if (i + 1) % 2000 == 0:
            logger.info("  Preprocessed %d / %d documents …", i + 1, len(documents))
    logger.info("Preprocessing complete. %d documents processed.", len(cleaned))
    return cleaned


# ── Sklearn-compatible transformer ───────────────────────────────────────────

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer wrapping :func:`clean_corpus`.

    Can be used as the first step of an ``sklearn.pipeline.Pipeline``.

    Parameters
    ----------
    lowercase : bool
    remove_stopwords : bool
    lemmatize : bool
    min_token_len : int
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_token_len: int = 3,
    ) -> None:
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_len = min_token_len

    def fit(self, X, y=None):  # noqa: N803
        return self                           # stateless transformer

    def transform(self, X, y=None):  # noqa: N803
        return clean_corpus(
            list(X),
            lowercase=self.lowercase,
            remove_stopwords=self.remove_stopwords,
            lemmatize=self.lemmatize,
            min_token_len=self.min_token_len,
        )
