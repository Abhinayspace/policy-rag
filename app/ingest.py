"""
ingest.py — PDF ingestion pipeline.

Loads all PDFs from /pdfs, chunks them, embeds each chunk using nomic-embed-text,
and stores vectors in a persistent ChromaDB collection.

Run once to build the index, and re-run whenever PDFs change.
IMPORTANT: Delete vector_store/ before re-running to avoid stale data.

Usage:
    python -m app.ingest
"""

import logging
import chromadb

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import get_embedding_model

logging.basicConfig(level=logging.WARNING)


def run():
    # ── 1. Load PDFs ──────────────────────────────────────────────────────────
    print("Loading PDFs...")
    docs = SimpleDirectoryReader(
        "pdfs/",
        required_exts=[".pdf"],
        file_extractor={".pdf": PDFReader()},
    ).load_data()
    print(f"  Loaded {len(docs)} pages across all PDFs")

    if len(docs) == 0:
        raise RuntimeError(
            "No pages loaded. Check that /pdfs contains text-based PDFs "
            "(not scanned images). Verify by selecting text in a PDF viewer."
        )

    # ── 2. Configure chunking ─────────────────────────────────────────────────
    # 768 tokens preserves full clauses in dense policy text.
    # 100 token overlap prevents context loss at chunk boundaries.
    splitter = SentenceSplitter(chunk_size=768, chunk_overlap=100)

    # ── 3. Connect to ChromaDB ────────────────────────────────────────────────
    # PersistentClient means data survives between runs.
    # get_or_create_collection is safe to call on every run — won't crash if collection exists.
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path="vector_store/")
    collection = client.get_or_create_collection("policy_docs")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ── 4. Build index: chunk → embed → store ────────────────────────────────
    print("Chunking, embedding, and storing vectors...")
    print("  (This takes a few minutes on first run — Ollama is embedding locally)")
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=get_embedding_model(),
        transformations=[splitter],
        show_progress=True,
    )

    # ── 5. Verify ─────────────────────────────────────────────────────────────
    count = collection.count()
    print(f"\nDone. ChromaDB collection 'policy_docs' contains {count} chunks.")

    if count == 0:
        raise RuntimeError(
            "Collection is empty after ingestion. "
            "The embedding model likely failed silently — check that Ollama is running "
            "and nomic-embed-text is pulled: ollama pull nomic-embed-text"
        )

    # ── 6. Metadata spot-check ────────────────────────────────────────────────
    sample_nodes = list(index.docstore.docs.values())
    if sample_nodes:
        sample = sample_nodes[0]
        print(f"\nSample chunk metadata: {sample.metadata}")
        print(f"Sample chunk text (first 200 chars):\n  {sample.text[:200]}")

    return index


if __name__ == "__main__":
    run()
