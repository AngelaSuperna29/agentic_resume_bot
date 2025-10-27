# src/ingestion.py
import os
import json
from pathlib import Path
import pdfplumber
import docx

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
RESUME_TEXT_DIR = DATA_DIR / 'resumes_text'
RAW_RESUME_DIR = DATA_DIR / 'resumes'
RESUME_TEXT_DIR.mkdir(parents=True, exist_ok=True)
RAW_RESUME_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(path):
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            text_parts.append(p.extract_text() or "")
    return "\n".join(text_parts)

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def ingest_resume(filepath, candidate_id=None, name=None, metadata=None):
    """Ingests one resume file and writes a cleaned text JSON file.

    Returns path to JSON file with fields: id, name, raw_text, metadata
    """
    filepath = Path(filepath)
    if candidate_id is None:
        candidate_id = filepath.stem
    if name is None:
        name = filepath.stem

    ext = filepath.suffix.lower()
    if ext == '.pdf':
        text = extract_text_from_pdf(filepath)
    elif ext in ('.docx', '.doc'):
        text = extract_text_from_docx(filepath)
    else:
        text = filepath.read_text(encoding='utf-8', errors='ignore')

    # basic cleaning
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    out = {
        'id': candidate_id,
        'name': name,
        'raw_text': text,
        'metadata': metadata or {}
    }
    out_path = RESUME_TEXT_DIR / f"{candidate_id}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out_path

def ingest_all_from_folder(folder_path):
    p = Path(folder_path)
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in ('.pdf', '.docx', '.txt'):
            print('Ingesting', f)
            ingest_resume(f)
