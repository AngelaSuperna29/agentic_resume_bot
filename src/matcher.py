import json
import os
import numpy as np
import faiss
from src.embeddings import embed_text
from src.ingestion import RESUME_TEXT_DIR
from src.indexer import INDEX_FILE


def load_index():
    """Load FAISS index if available."""
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError("❌ No FAISS index found. Run indexer.py first.")
    index = faiss.read_index(INDEX_FILE)
    return index

def load_resume_texts():
    """Load all resume text data from JSON files."""
    resumes = []
    files = []
    for fname in os.listdir(RESUME_TEXT_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(RESUME_TEXT_DIR, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                resumes.append(data["text"])
                files.append(fname.replace(".json", ""))
    return resumes, files

def match_job(job_text, top_k=5):
    """Match job descriptions to resumes using FAISS similarity search."""
    index = load_index()
    resumes, resume_files = load_resume_texts()

    # Embed the job description
    job_embedding = np.array([embed_text(job_text)], dtype="float32")

    # Search for nearest matches
    distances, indices = index.search(job_embedding, top_k)

    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(resumes):
            summary = resumes[idx][:400].replace("\n", " ") + "..."
            results.append((resume_files[idx], float(distance), summary))
    return results
