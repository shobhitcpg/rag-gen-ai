import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension: int =384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []
    def add_texts(self, vectors: np.ndarray, texts: list):
        #Normalize vectors to unit length for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.documents.extend(texts)
    def search_vc(self, query_vector: np.ndarray, top_k: int=5):
        #Normailize query vector
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        return [(self.documents[i], distances[0][idx]) for idx, i in enumerate(indices[0])] 