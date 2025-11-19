from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from embeddings_store import EmbeddingStore
from utils import MODEL_NAME

def make_documents_from_resumes(resume_texts):
    docs = []
    for text, meta in resume_texts:
        docs.append(Document(page_content=text, metadata=meta))
    return docs

class ResumeRAG:
    def __init__(self):
        self.llm = Ollama(model=MODEL_NAME)
        self.embed_store = EmbeddingStore()
        self.retriever = None

    def build(self, resume_texts):
        docs = make_documents_from_resumes(resume_texts)
        vs = self.embed_store.build_store(docs)
        self.retriever = vs.as_retriever(search_type='similarity', search_kwargs={'k': 5})
        self.chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type='stuff', retriever=self.retriever)

    def query(self, question):
        ans = self.chain.run(question)
        return {'answer': ans}

    def rank_candidates(self, job_desc, top_k=5):
        results = self.embed_store.vs.similarity_search_with_score(job_desc, k=top_k)
        ranked = []
        for doc, score in results:
            ranked.append({'metadata': doc.metadata, 'score': float(score), 'text_snippet': doc.page_content[:800]})
        return ranked
