"""src/preprocessor.py — Lightweight text cleaning for TF-IDF pipelines."""
import re, logging
import pandas as pd

logger = logging.getLogger(__name__)

_HTML   = re.compile(r"<[^>]+>")
_URL    = re.compile(r"https?://\S+|www\.\S+")
_WIRE   = re.compile(r"^\s*(AP|AFP|Reuters|REUTERS|UPI)\s*[-–—]?\s*", re.IGNORECASE)
_SPACE  = re.compile(r"\s+")
_SPEC   = re.compile(r"[^\w\s\.\,\!\?\;\:\'\"-]")
_ENTITY = re.compile(r"#\d+;")

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = _ENTITY.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _WIRE.sub("", text)
    text = _SPEC.sub(" ", text)
    text = _SPACE.sub(" ", text)
    return text.strip()

def clean_dataframe(df: pd.DataFrame, text_col: str = "text", inplace: bool = False) -> pd.DataFrame:
    if not inplace: df = df.copy()
    logger.info(f"Cleaning '{text_col}' ({len(df):,} rows)…")
    df["text_clean"] = df[text_col].apply(clean_text)
    empty = df["text_clean"].str.strip() == ""
    if empty.sum(): df = df[~empty]
    logger.info("Cleaning complete.")
    return df
