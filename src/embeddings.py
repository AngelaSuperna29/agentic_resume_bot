# src/embeddings.py
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import numpy as np

vectorizer = TfidfVectorizer(max_features=768)

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50):
    """Simple chunker by words. Returns list of text chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def embed_texts(texts: List[str]):
    """Return TF-IDF embeddings for text or list of texts."""
    if isinstance(texts, str):
        texts = [texts]
    embeddings = vectorizer.fit_transform(texts).toarray()
    return embeddings

# backward compatibility for old imports
embed_text = embed_texts
