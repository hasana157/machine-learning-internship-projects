"""
preprocessor.py
---------------
Text normalisation: lowercasing, punctuation removal, stopword removal,
and tokenisation. The output feeds directly into TF-IDF vectorisation.

Stopwords are bundled inline — no NLTK download required.
"""

import re
import string
import logging

logger = logging.getLogger(__name__)

# Built-in English stopwords (NLTK-equivalent, zero network dependency)
_STOPWORDS: set = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve",
    "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn",
}


def _get_stopwords() -> set:
    return _STOPWORDS


def clean_text(text: str) -> str:
    """
    Full text-cleaning pipeline applied to a single string.

    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove punctuation / digits
      4. Tokenise (whitespace split)
      5. Remove English stopwords
      6. Rejoin tokens
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # 4. Tokenise
    tokens = text.split()

    # 5. Stopword removal
    stop_words = _get_stopwords()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # 6. Rejoin
    return " ".join(tokens)


def preprocess_series(series):
    """Apply clean_text to a pandas Series; return cleaned Series."""
    logger.info("Preprocessing %d text samples...", len(series))
    cleaned = series.map(clean_text)
    logger.info("Preprocessing complete.")
    return cleaned
