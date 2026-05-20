import src

class RAGOrchestrator:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.embedding = src.Embedding(model_name=embedding_model)
        self.vector_store = src.VectorStore()
        self.retrieval_engine = src.RetrievalEngine(self.embedding, self.vector_store)

    def ingest_documents(self,file_path: str):
        # Placeholder for document ingestion logic
        with open(file_path, 'r') as f:
            documents =  [line.strip() for line in f if line.strip()]
        self.add_documents(documents)

    def add_documents(self, documents: list[str]):
        embeddings = self.embedding.embed(documents)
        self.vector_store.add_texts(embeddings, documents)

    def retrieve(self, query: str, top_k: int = 5):
        strategy_a_result = self.retrieval_engine.strategy_a(query)
        strategy_b_result = self.retrieval_engine.strategy_b(query)
        return {
            "strategy_a": strategy_a_result,
            "strategy_b": strategy_b_result
        }
if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "..", "data", "dataset.txt")

    orchestrator = RAGOrchestrator()
    orchestrator.ingest_documents(dataset_path)
    query = "What are the key features of peak load handling?"
    results = orchestrator.retrieve(query)
    print("Results from Strategy A:")
    print(results["strategy_a"])
    print("\nResults from Strategy B:")
    print(results["strategy_b"])