"""src/splitter.py — Stratified train/val split."""
import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def split_data(df, val_size=0.15, label_col="label", text_col="text_clean", random_state=42):
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df[label_col],
        random_state=random_state, shuffle=True,
    )
    logger.info(f"Split → Train: {len(train_df):,} | Val: {len(val_df):,} (val={val_size:.0%})")
    for name, d in [("Train", train_df), ("Val", val_df)]:
        dist = d[label_col].value_counts(normalize=True).to_dict()
        logger.info(f"  {name}: { {k: f'{v:.1%}' for k, v in dist.items()} }")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
