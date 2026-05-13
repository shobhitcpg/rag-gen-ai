import os
import dotenv
dotenv.load_dotenv()

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))
except ImportError:
    vertexai = None
    print("Vertex AI SDK not found. Using mock implementations for retrieval strategies.")

class RetrievalEngine:
    def __init__(self, embedder, vector_store):
        self.emb = embedder
        self.vs = vector_store
        self.use_mock = os.getenv("USE_MOCK", "True") == "True"
        if not self.use_mock and vertexai:
            self.model = GenerativeModel("gemini-1.5-flash")

    def add_documents(self, texts: list[str]):
        vectors = self.emb.embed(texts)
        self.vs.add_texts(vectors, texts)

    def query(self, query_text: str, top_k: int=5):
        query_vector = self.emb.embed([query_text])
        return self.vs.search_vc(query_vector, top_k)

    def strategy_a(self, query_text: str):
        #Example strategy: Retrieve top 5 relevant documents and concatenate them
        results = self.query(query_text, top_k=5)
        retrieved_docs = [doc for doc, score in results]
        return " ".join(retrieved_docs)

    def strategy_b(self, query_text: str):
        if self.use_mock:
            expanded_query = self.mock_query_expansion(query_text)
        else:
            # Use Vertex AI's query expansion capabilities
            expanded_query = self.vertex_query_expansion(query_text)
        results = self.query(expanded_query, top_k=5)
        retrieved_docs = [doc for doc, score in results]
        return " ".join(retrieved_docs)

    def mock_query_expansion(self, query_text: str):
        # For real case vertexAI would enhance this query
        #BuT here we just update with our own dummy text
        return f"Expanded query"

    def vertex_query_expansion(self, query_text: str):
        # Placeholder for actual Vertex AI query expansion logic
        prompt = (
        f"Act as a search engine optimizer. Expand the following user query "
        f"with synonyms and related technical terms to improve retrieval. "
        f"Output ONLY the expanded keywords, no conversational filler.\n\n"
        f"Query: {query_text}"
        )
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 50,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            )
            return response.text.strip() if response.text else query_text
        
        except Exception as e:
            print(f"Error during Vertex AI query expansion: {e}")
            return self.mock_query_expansion(query_text 
        )