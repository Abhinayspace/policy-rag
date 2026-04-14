"""
streamlit_app.py — Chat UI for the Policy RAG system.

Talks to the FastAPI backend at http://localhost:8000.
Start the API first: uvicorn app.main:app --port 8000
Then run this: streamlit run ui/streamlit_app.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Policy Assistant",
    page_icon="📋",
    layout="centered",
)

# ── Session state ─────────────────────────────────────────────────────────────
# Persists chat history across rerenders within the same browser session.
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📋 Policy Assistant")
    st.caption("Ask questions about company policies.")
    st.divider()

    # Health check — shows current LLM mode
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API connected — LLM: `{health.get('llm_mode', '?')}`")
    except Exception:
        st.error("API not reachable. Start the server:\n`uvicorn app.main:app --port 8000`")

    st.divider()

    # Re-index button
    st.subheader("Document Index")
    st.caption("Add new PDFs to /pdfs and re-index to include them.")
    if st.button("🔄 Re-index PDFs", use_container_width=True):
        with st.spinner("Re-indexing... this takes a few minutes."):
            try:
                resp = requests.post(f"{API_URL}/ingest", timeout=600)
                data = resp.json()
                if resp.status_code == 200:
                    st.success(f"Done! {data['chunk_count']} chunks indexed.")
                else:
                    st.error(f"Error: {data.get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
st.header("Policy Q&A")

# Render full conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Show source citations under assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    fname = src["file_name"].replace("_", " ").replace(".pdf", "")
                    page = src.get("page", "?")
                    score = src.get("score")
                    score_str = f" (relevance: {score:.2f})" if score else ""
                    st.caption(f"• **{fname}** — page {page}{score_str}")

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about a company policy..."):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Call API and stream answer
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt},
                    timeout=400,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data["sources"]
                else:
                    answer = f"Error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}"
                    sources = []
            except requests.exceptions.Timeout:
                answer = "The query timed out. The model is processing — try again or ask a simpler question."
                sources = []
            except Exception as e:
                answer = f"Could not reach the API: {e}"
                sources = []

        st.write(answer)

        # Show sources
        if sources:
            with st.expander("📄 Sources", expanded=False):
                for src in sources:
                    fname = src["file_name"].replace("_", " ").replace(".pdf", "")
                    page = src.get("page", "?")
                    score = src.get("score")
                    score_str = f" (relevance: {score:.2f})" if score else ""
                    st.caption(f"• **{fname}** — page {page}{score_str}")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
