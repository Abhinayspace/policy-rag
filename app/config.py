"""
config.py — the LLM swap point.

Every other file in this project imports get_llm() and get_embedding_model()
from here. No other file ever names a model directly.

Switch models by changing LLM_MODE in .env:
  ollama  → local Llama3.2 via Ollama (default, free, no API key needed)
  claude  → Anthropic Claude Sonnet (requires ANTHROPIC_API_KEY)
  gemini  → Google Gemini Flash (requires GOOGLE_API_KEY)

Embeddings always stay local (nomic-embed-text via Ollama) regardless of LLM_MODE.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """Return the configured LLM based on LLM_MODE env var."""
    mode = os.getenv("LLM_MODE", "ollama").lower()

    if mode == "ollama":
        from llama_index.llms.ollama import Ollama
        return Ollama(model="llama3.2", request_timeout=120.0)

    if mode == "claude":
        from llama_index.llms.anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in .env")
        return Anthropic(model="claude-sonnet-4-20250514", api_key=api_key)

    if mode == "gemini":
        from llama_index.llms.gemini import Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in .env")
        return Gemini(model="models/gemini-2.0-flash", api_key=api_key)

    raise ValueError(f"Unknown LLM_MODE '{mode}'. Choose: ollama | claude | gemini")


def get_embedding_model():
    """Return the embedding model. Always local — never pays for an embedding API."""
    from llama_index.embeddings.ollama import OllamaEmbedding
    return OllamaEmbedding(model_name="nomic-embed-text")
