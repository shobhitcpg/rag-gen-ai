# GenAI RAG Assessment

This project demonstrates a Python-based implementation of a **Retrieval-Augmented Generation (RAG)** pipeline, capable of running entirely locally using mock implementations or connecting to Google Vertex AI. 

The primary goal is to demonstrate:
- **Vector Embeddings**: Creating sentence embeddings using `sentence-transformers`.
- **Vector Storage**: Performing highly efficient similarity searches over an in-memory vector store powered by `FAISS`.
- **Query Enhancement (Strategy B)**: Simulating a Google Vertex AI-like pre-processing step where a user's query is expanded with synonyms or technical context *before* the similarity search to yield much better search results.
- **Direct Retrieval (Strategy A)**: Performing a traditional RAG search using the raw user query.

## Project Structure & Flow

```text
genai-rag-assessment/
├── data/
│   └── dataset.txt        # Sample text knowledge base to be ingested
├── src/
│   ├── orchestrator.py    # Main entry point. Handles end-to-end ingestion and query routing.
│   ├── retrieval.py       # Core retrieval logic (Strategy A vs. Strategy B) & Query Expansion.
│   ├── embedding.py       # Wrapper around sentence-transformers to convert text to vectors.
│   ├── storage.py         # The FAISS-based vector database layer.
│   └── sample.ipynb       # Jupyter notebook showing interactive testing.
├── tests/
│   └── test_pipeline.py   # Pytest suite ensuring the mock and vertex logic is bulletproof.
├── pyproject.toml         # Dependency configurations
└── .env.example           # Example environment variables (for Vertex API keys and mock toggles)
```

### How Data Flows Through the System
1. **Ingestion**: `RAGOrchestrator` (`src/orchestrator.py`) reads the documents from `data/dataset.txt`.
2. **Embedding**: Documents are sent to `src/embedding.py` and converted into dense vector representations.
3. **Storage**: Vectors and their raw text are stored in the FAISS index (`src/storage.py`).
4. **Retrieval**: When a query is asked, it goes through `src/retrieval.py`:
   - **Strategy A**: Directly queries the vector store for the top 5 nearest matches using the raw query.
   - **Strategy B**: First passes the query through an LLM (Vertex AI or a Mock function) to expand its context, *then* performs the similarity search on the expanded query.

## Setup Guide (Windows / uv)

This project relies on `uv`, an extremely fast Python package and project manager written in Rust.

### 1. Install uv
If you don't already have `uv` installed on Windows, open your PowerShell and run:
```powershell
pip install uv
```
*(Alternatively, you can install it standalone via: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`)*

### 2. Setup the Environment & Dependencies
Navigate to the root of your project directory and execute:
```powershell
# This will automatically read pyproject.toml, create a .venv, and install all dependencies (like FAISS, Pytest, Sentence-Transformers)
uv sync
```

### 3. Activate the Virtual Environment
Before running any code, you must activate the isolated environment:
```powershell
.venv\Scripts\activate
```

### 4. Configure Environment Variables
Copy the `.env.example` file to create a `.env` file:
```powershell
cp .env.example .env
```
Inside `.env`, you can toggle `USE_MOCK=True` (for local development without Vertex API keys) or set it to `False` and supply your actual `PROJECT_ID` and `LOCATION` to use the real Google Vertex AI models.

## Running the Application

To run the full end-to-end flow and see the difference between a raw search (Strategy A) and an expanded context search (Strategy B):

```powershell
python -m src.orchestrator
```

## Running Tests

We have a comprehensive test suite built with `pytest` to ensure the mock logic and Vertex branches work seamlessly.

To run the tests:
```powershell
pytest tests/
```
If you wish to see debugging output and detailed function flow (such as `print` statements during tests), use the verbose flags:
```powershell
pytest -s -v tests/
```
