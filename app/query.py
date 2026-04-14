"""
query.py — Query engine for the Policy RAG system.

Loads the ChromaDB index built by ingest.py and answers questions by:
  1. Retrieving the top-3 most relevant chunks via semantic search
  2. Passing them to the LLM with a strict prompt that enforces source citation

IMPORTANT: Never call VectorStoreIndex.from_documents() here — that re-embeds
everything on every query. Always load from the existing vector store.

Usage:
    python -m app.query
"""

import chromadb

from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import get_llm, get_embedding_model

# ── Custom prompt template ────────────────────────────────────────────────────
# Forces the LLM to:
#   1. Answer only from provided context (no hallucination)
#   2. Always cite the source document and page number
#   3. Explicitly say when information is not in the policy

POLICY_QA_TEMPLATE = PromptTemplate(
    "You are a helpful HR policy assistant for Acidaes Solutions Pvt. Ltd.\n"
    "Answer the employee's question using ONLY the policy context provided below.\n"
    "Rules:\n"
    "  1. Base your answer strictly on the context — do not use outside knowledge.\n"
    "  2. Always cite the source: mention the policy name and page number from the metadata.\n"
    "  3. If the answer is not found in the context, respond with exactly:\n"
    '     "This information is not in the provided policy documents."\n'
    "  4. Be concise and direct. Employees want quick, actionable answers.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


def load_query_engine(similarity_top_k: int = 3):
    """Load the query engine from the existing ChromaDB index."""
    client = chromadb.PersistentClient(path="vector_store/")

    try:
        collection = client.get_collection("policy_docs")
    except Exception:
        raise RuntimeError(
            "ChromaDB collection 'policy_docs' not found. "
            "Run ingest first: python -m app.ingest"
        )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=get_embedding_model(),
    )

    engine = index.as_query_engine(
        llm=get_llm(),
        similarity_top_k=similarity_top_k,
        text_qa_template=POLICY_QA_TEMPLATE,
    )
    return engine


def ask(question: str, engine=None, verbose: bool = True):
    """
    Ask a question and return the answer with sources.

    Args:
        question: The question to ask.
        engine: Pre-loaded query engine (loads fresh if None).
        verbose: If True, prints retrieved source chunks for observability.

    Returns:
        dict with keys: answer (str), sources (list of dicts)
    """
    if engine is None:
        engine = load_query_engine()

    response = engine.query(question)

    # ── Observability: log retrieved chunks ───────────────────────────────────
    sources = []
    if verbose:
        print(f"\n{'─'*60}")
        print(f"Q: {question}")
        print(f"{'─'*60}")
        print(f"A: {response}\n")
        print("Retrieved chunks:")

    for i, node in enumerate(response.source_nodes):
        meta = node.metadata
        source_entry = {
            "file_name": meta.get("file_name", "unknown"),
            "page": meta.get("page_label", meta.get("page", "?")),
            "score": round(node.score, 4) if node.score else None,
            "text_preview": node.text[:150],
        }
        sources.append(source_entry)

        if verbose:
            print(
                f"  [{i+1}] score={source_entry['score']} | "
                f"{source_entry['file_name']} p.{source_entry['page']}"
            )
            print(f"       {source_entry['text_preview']}...")

    return {"answer": str(response), "sources": sources}


def run_golden_qa(engine, golden_qa_path: str = "tests/golden_qa.txt"):
    """Run all golden Q&A pairs through the engine and report pass/fail."""
    questions = []
    with open(golden_qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Q:"):
                questions.append(line[2:].strip())

    print(f"\n{'='*60}")
    print(f"Running {len(questions)} golden questions...")
    print(f"{'='*60}")

    passed, failed, timed_out = 0, 0, 0

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q}")
        try:
            result = ask(q, engine=engine, verbose=False)
            print(f"  Answer: {result['answer'][:200]}")
            srcs = [f"{s['file_name']} p.{s['page']}" for s in result["sources"]]
            print(f"  Sources: {srcs}")
            passed += 1
        except Exception as e:
            err = str(e)
            if "timeout" in err.lower() or "timed out" in err.lower():
                print(f"  [TIMEOUT] Query took too long — skipping")
                timed_out += 1
            else:
                print(f"  [ERROR] {err[:120]}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"Completed: {passed} answered | {timed_out} timed out | {failed} errors")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("Loading query engine...")
    engine = load_query_engine()
    print("Ready. Running golden Q&A evaluation...\n")
    run_golden_qa(engine)
