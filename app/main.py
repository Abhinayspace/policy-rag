"""
main.py — FastAPI backend for the Policy RAG system.

Keeps this layer thin: HTTP concerns only.
All AI logic lives in query.py and ingest.py — never duplicate it here.

Routes:
    GET  /health  → server + config status
    POST /query   → ask a question, get answer + sources
    POST /ingest  → re-index all PDFs in /pdfs

Run:
    uvicorn app.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs  (auto-generated Swagger UI)
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="Policy RAG API",
    description="Ask questions against company policy documents.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Required so Streamlit (port 8501) can call this API (port 8000).
# Without this, browser fetch calls are blocked by CORS policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-load the query engine once at startup ─────────────────────────────────
# Loading it per-request would re-connect to ChromaDB on every call — wasteful.
_query_engine = None


def get_engine():
    global _query_engine
    if _query_engine is None:
        from app.query import load_query_engine
        _query_engine = load_query_engine()
    return _query_engine


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    file_name: str
    page: str
    score: float | None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class IngestResponse(BaseModel):
    status: str
    chunk_count: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check that the server is running and show current LLM mode."""
    return {
        "status": "ok",
        "llm_mode": os.getenv("LLM_MODE", "ollama"),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Ask a question against the policy documents.

    Returns the answer and the source chunks used to generate it.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        from app.query import ask
        result = ask(request.question, engine=get_engine(), verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=result["answer"],
        sources=[
            SourceItem(
                file_name=s["file_name"],
                page=str(s["page"]),
                score=s.get("score"),
            )
            for s in result["sources"]
        ],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    Re-index all PDFs in the /pdfs folder.

    Deletes the existing ChromaDB collection and rebuilds from scratch.
    Use this whenever PDFs are added or updated.
    """
    try:
        import chromadb
        import shutil

        # Clear existing vector store so re-ingest is clean
        if os.path.exists("vector_store/"):
            shutil.rmtree("vector_store/")

        from app.ingest import run
        # Reset the cached query engine so next query reloads the fresh index
        global _query_engine
        _query_engine = None

        run()

        # Return chunk count from fresh collection
        client = chromadb.PersistentClient(path="vector_store/")
        count = client.get_collection("policy_docs").count()

        return IngestResponse(status="done", chunk_count=count)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
