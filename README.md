## 📌 Executive Summary

DocuChat is an end-to-end Retrieval-Augmented Generation (RAG) system engineered for high-throughput document Q&A. Instead of piping raw multi-page text into bloated context windows, DocuChat indexes partitioned document chunks locally and uses cosine vector search to supply an ultra-low-latency Groq-hosted LLM with targeted, relevant context.

## 🏗️ Architecture & Data Flow

┌──────────────┐
│  User PDF    │
└──────┬───────┘
│
▼
┌─────────────────────────┐
│  document_processor.py  │  ➔ Page-aware text extraction (pypdf)
│                         │  ➔ Sliding-window chunking (~350 words, 50-word overlap)
└─────────┬───────────────┘
│
▼
┌─────────────────────────┐
│     vector_store.py     │  ➔ Local embedding generation (sentence-transformers)
│                         │  ➔ In-memory FAISS/NumPy vector similarity store
└─────────┬───────────────┘
│
│ Top-k Chunks + System Grounding Constraints
▼
┌─────────────────────────┐
│      llm_client.py      │  ➔ Groq LPU API (openai/gpt-oss-20b / openai/gpt-oss-120b)
└─────────┬───────────────┘
│
▼
┌─────────────────────────┐
│         app.py          │  ➔ Streamlit UI with expandable citation badges
└─────────────────────────┘