import pandas as pd

def load_raw_data(path="../data/raw/spam.tsv"):
    """
    Load the raw spam dataset and convert labels to 0/1.
    """
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["label", "text"]
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df