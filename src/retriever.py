# src/retriever.py
import faiss
import numpy as np
import json
from pathlib import Path
from src.embeddings import MODEL

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
INDEX_DIR = DATA_DIR / 'index'
INDEX_PATH = INDEX_DIR / 'faiss.index'
IDS_PATH = INDEX_DIR / 'ids.json'

class Retriever:
    def __init__(self):
        if not INDEX_PATH.exists():
            raise FileNotFoundError('Index not found. Run indexer.build_index() first')
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(IDS_PATH, 'r', encoding='utf-8') as fh:
            self.ids = json.load(fh)

    def search(self, query: str, k: int = 10):
        q_emb = MODEL.encode([query]).astype('float32')
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for rank_pos, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            meta = self.ids[idx]
            score = float(D[0][rank_pos])
            results.append({'score': score, 'meta': meta})
        return results

if __name__ == '__main__':
    r = Retriever()
    print(r.search('python machine learning', k=5))
