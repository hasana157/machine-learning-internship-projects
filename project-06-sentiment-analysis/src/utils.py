import re
import json
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK data is available
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(path, payload):
    """Save dictionary as JSON file."""
    ensure_dir(Path(path).parent)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)

def clean_text(text, remove_stopwords=True, use_stemming=True):
    """
    Clean and preprocess text data.
    - Lowercase
    - Remove HTML tags
    - Remove punctuation and numbers
    - Remove extra whitespace
    - Remove stopwords (optional)
    - Apply Porter stemming (optional)
    """
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    
    # Stemming
    if use_stemming:
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]
    
    return ' '.join(words)