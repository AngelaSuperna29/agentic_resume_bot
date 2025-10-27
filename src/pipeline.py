# src/pipeline.py
from src.retriever import Retriever
from src.summarizer import summarize_candidate
from collections import defaultdict

class MatchPipeline:
    def __init__(self):
        self.retriever = Retriever()

    def match_job(self, job_text: str, top_k: int = 200, shortlist_n: int = 5):
        # 1) retrieve many chunks that match the job description
        hits = self.retriever.search(job_text, k=top_k)

        # 2) group hits by resume_id
        grouped = defaultdict(list)
        for h in hits:
            meta = h['meta']
            grouped[meta['resume_id']].append(meta['text'])

        # 3) summarize each resume vs job
        scored = []
        for resume_id, chunks in grouped.items():
            summary = summarize_candidate(chunks, job_text)
            summary['resume_id'] = resume_id
            scored.append(summary)

        # 4) sort by score
        scored_sorted = sorted(scored, key=lambda x: x.get('score', 0), reverse=True)
        return scored_sorted[:shortlist_n]

if __name__ == '__main__':
    mp = MatchPipeline()
    job = 'Looking for a Data Scientist with Python, pytorch, transformers, 3+ years experience in NLP.'
    print(mp.match_job(job))
