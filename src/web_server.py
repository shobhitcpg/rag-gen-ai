import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.orchestrator import RAGOrchestrator

# Initialize FastAPI App
app = FastAPI(
    title="GenAI RAG Assessment Dashboard",
    description="Interactive evaluation environment for RAG Retrieval Strategies",
    version="1.0.0"
)

# Setup paths dynamically relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
dataset_path = os.path.join(current_dir, "..", "data", "dataset.txt")

# Ensure static files directory exists
os.makedirs(static_dir, exist_ok=True)

# Initialize RAG Orchestrator on Startup
orchestrator = RAGOrchestrator()
try:
    if os.path.exists(dataset_path):
        orchestrator.ingest_documents(dataset_path)
        print(f"Successfully ingested default dataset: {dataset_path}")
    else:
        print(f"Dataset path not found: {dataset_path}. Starting with empty vector store.")
except Exception as e:
    print(f"Failed to ingest default documents: {e}")

# Pydantic schemas for API inputs
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# REST API Endpoints

@app.get("/api/documents")
async def get_documents():
    """Retrieve all ingested documents in the local vector store."""
    try:
        # Get the unique list of documents stored in orchestrator vector store
        docs = orchestrator.vector_store.documents
        return {
            "total_documents": len(docs),
            "documents": docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def execute_query(request: QueryRequest):
    """
    Execute standard vector search (Strategy A) and expanded search (Strategy B),
    returning detailed retrieval matches, cosine similarity scores, and expanded queries.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 1. Strategy A: Direct Retrieval
        # Retrieve raw vector search results from retrieval engine
        a_raw_results = orchestrator.retrieval_engine.query(request.query, top_k=request.top_k)
        strategy_a_matches = [
            {"text": doc, "score": round(float(score), 4)}
            for doc, score in a_raw_results
        ]

        # 2. Strategy B: Expanded Query Retrieval
        # Retrieve expanded query terms
        use_mock = orchestrator.retrieval_engine.use_mock
        if use_mock:
            expanded_query = orchestrator.retrieval_engine.mock_query_expansion(request.query)
        else:
            expanded_query = orchestrator.retrieval_engine.vertex_query_expansion(request.query)

        # Retrieve vector search results using the expanded query terms
        b_raw_results = orchestrator.retrieval_engine.query(expanded_query, top_k=request.top_k)
        strategy_b_matches = [
            {"text": doc, "score": round(float(score), 4)}
            for doc, score in b_raw_results
        ]

        return {
            "query": request.query,
            "top_k": request.top_k,
            "strategy_a": {
                "results": strategy_a_matches
            },
            "strategy_b": {
                "expanded_query": expanded_query,
                "results": strategy_b_matches
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static folder and serve index.html
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    else:
        return {
            "message": "FastAPI server running successfully. Static files not found.",
            "api_endpoints": ["/api/documents", "/api/query"]
        }

if __name__ == "__main__":
    print("Starting FastAPI development server at http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
