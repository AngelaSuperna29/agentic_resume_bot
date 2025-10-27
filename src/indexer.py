# src/indexer.py
import os
import faiss
import numpy as np
import json
from pathlib import Path
from src.embeddings import chunk_text, embed_texts

# Define where to store the FAISS and metadata files
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
RESUME_TEXT_DIR = DATA_DIR / 'resumes_text'
INDEX_DIR = DATA_DIR / 'index'
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Path to store a serialized index (used in matcher)
INDEX_FILE = os.path.join("data", "index.pkl")

def build_index(recompute: bool = False):
    """Reads all resume JSONs, chunk them, embed, and build a FAISS index.

    Saves:
    - data/index/faiss.index
    - data/index/ids.json  (list of dicts: {id, resume_id, chunk_index, text})
    - data/index/embs.npy
    """
    ids_path = INDEX_DIR / 'ids.json'
    index_path = INDEX_DIR / 'faiss.index'
    embs_path = INDEX_DIR / 'embs.npy'

    # gather chunks
    items = []
    for f in RESUME_TEXT_DIR.glob('*.json'):
        with open(f, 'r', encoding='utf-8') as fh:
            doc = json.load(fh)
        resume_id = doc['id']
        text = doc['raw_text']
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            items.append({
                'id': f"{resume_id}_chunk_{i}",
                'resume_id': resume_id,
                'chunk_index': i,
                'text': c
            })

    texts = [it['text'] for it in items]
    if len(texts) == 0:
        raise ValueError('No resume texts found. Run ingestion first and put JSONs in data/resumes_text')

    # embeddings
    embs = embed_texts(texts).astype('float32')

    # normalize for cosine similarity (inner product)
    faiss.normalize_L2(embs)

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)

    faiss.write_index(index, str(index_path))
    np.save(embs_path, embs)
    with open(ids_path, 'w', encoding='utf-8') as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)

    print('✅ Index built:', index.ntotal, 'vectors')

if __name__ == '__main__':
    build_index()
