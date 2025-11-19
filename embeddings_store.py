from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

class EmbeddingStore:
    def __init__(self):
        self.embed = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.vs = None

    def build_store(self, docs):
        self.vs = FAISS.from_documents(docs, self.embed)
        return self.vs

    def similarity_search(self, query, k=5):
        if not self.vs:
            raise RuntimeError('Vector store not initialized')
        return self.vs.similarity_search(query, k=k)
