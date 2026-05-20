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
        # Local keyword mapping representing common topics in the dataset to simulate LLM query expansion
        synonyms = {
            "peak load": ["surge", "demand", "traffic spike", "resource saturation", "high load"],
            "spike": ["peak load", "surge", "saturation", "overload"],
            "overload": ["peak load", "surge", "cpu usage", "resource exhaustion"],
            "monitoring": ["metrics", "cpu usage", "memory consumption", "alerts", "telemetry"],
            "autoscaling": ["scale", "elasticity", "containers", "instances", "virtual machines"],
            "load distribution": ["load balancer", "traffic routing", "health check", "servers"],
            "balancer": ["load distribution", "routing", "servers", "nodes"],
            "caching": ["memory cache", "redis", "fast response", "session store", "latency"],
            "cache": ["caching", "memory", "in-memory", "latency", "redis"],
            "rate limiting": ["request control", "blocking", "throttling", "ddos protection"],
            "queueing": ["message queue", "decouple", "workers", "backend processing", "asynchronous"],
            "queue": ["queueing", "decoupling", "decouple", "workers", "async"],
            "database": ["indexes", "read replicas", "query optimization", "sql", "datastore"],
            "optimization": ["indexing", "performance tuning", "bottlenecks", "speed"],
            "graceful degradation": ["noncritical", "fallback", "redundancy", "resilience"],
            "testing": ["stress tests", "load tests", "simulated traffic", "validation", "benchmark"],
            "redundancy": ["failover", "backups", "high availability", "outages", "fault tolerance"],
            "cdn": ["content delivery network", "static assets", "offload", "edge caching"]
        }
        
        normalized = query_text.lower()
        expanded_terms = [query_text] # Keep original query terms
        
        for key, value in synonyms.items():
            if key in normalized:
                expanded_terms.extend(value)
                
        # Deduplicate terms while preserving order
        seen = set()
        deduped = []
        for term in " ".join(expanded_terms).split():
            clean = term.strip(",.?!()\"'").lower()
            if clean and clean not in seen:
                seen.add(clean)
                deduped.append(clean)
                
        # Return the expanded query as a single string
        return " ".join(deduped)

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