import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def add_clean_column(df):
    """
    Add cleaned text and length column to DataFrame
    """
    df['clean_text'] = df['text'].apply(clean_text)
    df['length'] = df['clean_text'].apply(len)
    return df