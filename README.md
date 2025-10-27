# Agentic Resume Bot - Minimal Runnable Prototype

Quickstart:
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Put resumes (pdf/docx/txt) into `data/resumes/`.
3. Ingest them (or use the upload endpoint).
4. Build index:
   ```
   python -m src.indexer
   ```
5. Run the API:
   ```
   python -m src.app
   ```
6. POST `/match_job` with form field `job_text` to get shortlisted candidates.

Notes:
- Set OPENAI_API_KEY if you want real LLM summaries.
- This is a starting prototype. Add PII handling, authentication, and production hardening for real use.
