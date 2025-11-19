import io
from pdfminer.high_level import extract_text
from docx import Document

def parse_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def parse_pdf(path: str) -> str:
    return extract_text(path)

def parse_docx(path: str) -> str:
    doc = Document(path)
    return '\n'.join(p.text for p in doc.paragraphs)

def parse_resume(path: str) -> str:
    path = str(path)
    if path.lower().endswith('.pdf'):
        return parse_pdf(path)
    elif path.lower().endswith('.docx'):
        return parse_docx(path)
    else:
        return parse_txt(path)
