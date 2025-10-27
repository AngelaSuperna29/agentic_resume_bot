# src/summarizer.py
import os
import openai
import json
from typing import List

# Set OPENAI_API_KEY as env var or replace below (not recommended)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    print('Warning: OPENAI_API_KEY not found in environment. Set it before calling the summarizer.')
else:
    openai.api_key = OPENAI_API_KEY

def call_llm_system(prompt: str, model: str = 'gpt-4o'):
    """Simple wrapper around OpenAI completion. Replace with your provider specifics."""
    if OPENAI_API_KEY is None:
        # For offline dev: return a dummy JSON
        return json.dumps({'score': 50, 'headline': 'Candidate (demo)', 'top_skills': [], 'short_summary': 'Demo mode — no API key.'})

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant that summarizes candidate resumes and scores suitability for jobs."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=400
    )
    return resp['choices'][0]['message']['content']

def summarize_candidate(chunks: List[str], job_description: str):
    """Aggregates chunks and asks the LLM to produce structured summary + score.

    Returns dict with keys: score (0-100), headline, top_skills, short_summary, strengths, gaps
    """
    # take top N chars to keep prompt short
    joined = '\n\n'.join(chunks)[:6000]
    prompt = f"""
Resume excerpts:
{joined}

Job description:
{job_description}

Task: Produce JSON with fields:
- score: integer 0-100 (how suitable this candidate is for the job)
- headline: one-line headline (max 10 words)
- top_skills: list of strings (max 8)
- short_summary: 2-3 sentence summary
- strengths: list
- gaps: list

Return only JSON.
"""
    out = call_llm_system(prompt)
    try:
        parsed = json.loads(out)
    except Exception:
        # if the model returned text, try to extract JSON block
        import re
        m = re.search(r"\{.*\}", out, re.S)
        if m:
            parsed = json.loads(m.group(0))
        else:
            parsed = {'score': 50, 'headline': 'Could not parse LLM output', 'top_skills': [], 'short_summary': out, 'strengths': [], 'gaps': []}
    return parsed
