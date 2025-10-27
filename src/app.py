# src/app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
from src import ingestion
from src.indexer import build_index
from src.pipeline import MatchPipeline

app = FastAPI()
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
RAW_RESUME_DIR = DATA_DIR / 'resumes'

pipeline = None

@app.post('/upload_resume')
async def upload_resume(file: UploadFile = File(...), name: str = Form(None)):
    RAW_RESUME_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_RESUME_DIR / file.filename
    with open(target, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    # ingest the newly uploaded file
    ingestion.ingest_resume(target, name=(name or file.filename))
    return JSONResponse({'status': 'ok', 'filename': file.filename})

@app.post('/reindex')
async def reindex():
    build_index()
    return JSONResponse({'status': 'index_built'})

@app.post('/match_job')
async def match_job(job_text: str = Form(...), top_k: int = Form(200), shortlist_n: int = Form(5)):
    global pipeline
    if pipeline is None:
        pipeline = MatchPipeline()
    results = pipeline.match_job(job_text, top_k=int(top_k), shortlist_n=int(shortlist_n))
    return JSONResponse({'status': 'ok', 'results': results})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
