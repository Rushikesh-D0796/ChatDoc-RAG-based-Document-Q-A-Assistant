"""DocuChat — RAG-based Document Q&A Assistant.

Upload a PDF, ask questions about it, and get grounded answers with page citations.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from document_processor import extract_text_by_page, chunk_pages
from vector_store import VectorStore
from llm_client import ask_llm

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    try:
        groq_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        groq_key = None


st.set_page_config(page_title="Doc-Chat", page_icon="📄", layout="wide")

st.title("📄 Doc-Chat")
st.caption("Ask questions about your PDF and get grounded, source-cited answers.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_name" not in st.session_state:
    st.session_state.file_name = None

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None and uploaded_file.name != st.session_state.file_name:
        with st.spinner("Reading and indexing your document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            pages = extract_text_by_page(tmp_path)
            chunks = chunk_pages(pages)

            store = VectorStore()
            store.build(chunks)

            st.session_state.vector_store = store
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = []

            os.remove(tmp_path)

        st.success(f"Indexed {len(chunks)} chunks from {len(pages)} pages.")

    if st.session_state.vector_store is not None:
        st.info(f"Currently loaded: **{st.session_state.file_name}**")
        top_k = st.slider("Chunks to retrieve per question", 2, 8, 4)
    else:
        top_k = 4

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("View sources"):
                for src in msg["sources"]:
                    st.markdown(f"**Page {src['page']}** (similarity: {src['score']:.2f})")
                    st.caption(src["text"][:300] + "...")

question = st.chat_input("Ask a question about your document...")

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retrieved = st.session_state.vector_store.search(question, top_k=top_k)
                try:
                    answer = ask_llm(question, retrieved)
                except Exception as e:
                    answer = f"Error calling the LLM: {e}"

                st.write(answer)
                with st.expander("View sources"):
                    for src in retrieved:
                        st.markdown(f"**Page {src['page']}** (similarity: {src['score']:.2f})")
                        st.caption(src["text"][:300] + "...")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved
        })
