# policy-rag

A local RAG system for querying policy documents using LlamaIndex + ChromaDB.

## Stack
- LlamaIndex 0.10.x
- ChromaDB (persistent local vector store)
- Ollama (local LLM + embeddings by default)
- FastAPI + Streamlit (UI)

## Setup
```bash
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env            # fill in API keys if using Claude/Gemini
python -m app.ingest            # Phase 3
python -m app.query             # Phase 4
```

