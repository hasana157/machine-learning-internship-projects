"""src/data_loader.py — Unified data loading for AG News."""
import logging, re
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAP: Dict[int, str] = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

def load_from_csv(path: str, label_map: Optional[Dict]=None) -> pd.DataFrame:
    label_map = label_map or LABEL_MAP
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    logger.info(f"Loading CSV → {path}")
    df = pd.read_csv(path, encoding="utf-8")
    df["label"] = df["Class Index"].map(label_map)
    df = df.dropna(subset=["label"])
    df["text"] = df["Title"].str.strip() + ". " + df["Description"].str.strip()
    logger.info(f"Loaded {len(df):,} rows | {df['label'].value_counts().to_dict()}")
    return df[["text", "label", "Class Index"]]

def load_from_huggingface(split: str = "train", label_map: Optional[Dict]=None) -> pd.DataFrame:
    label_map = label_map or LABEL_MAP
    from datasets import load_dataset
    logger.info(f"Fetching AG News ({split}) from HuggingFace…")
    ds = load_dataset("ag_news", split=split)
    df = ds.to_pandas()
    df["Class Index"] = df["label"] + 1
    df["label"] = df["Class Index"].map(label_map)
    logger.info(f"Loaded {len(df):,} rows")
    return df[["text", "label", "Class Index"]]

def load_data(train_path=None, test_path=None, source="csv", label_map=None):
    if source == "csv":
        return load_from_csv(train_path, label_map), load_from_csv(test_path, label_map)
    return load_from_huggingface("train", label_map), load_from_huggingface("test", label_map)

# ── Internal helpers exposed for testing ──────────────────────────────────────
def _apply_label_map(df: pd.DataFrame, label_map: Dict[int, str]) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["Class Index"].map(label_map)
    return df.dropna(subset=["label"])

def _combine_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["Title"].str.strip() + ". " + df["Description"].str.strip()
    return df
