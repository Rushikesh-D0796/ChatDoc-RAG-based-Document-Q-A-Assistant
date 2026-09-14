"""Calls the Groq LLM API to generate answers grounded in retrieved context."""

import os
from groq import Groq

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
provided context from a document.

Rules:
- If the answer is not contained in the context, say "I don't have enough \
information in this document to answer that."
- Never make up facts that aren't in the context.
- Keep answers concise and directly relevant to the question.
"""


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    return Groq(api_key=api_key)


def build_context(chunks) -> str:
    parts = [f"[Page {c['page']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


def ask_llm(question: str, retrieved_chunks, model: str = "openai/gpt-oss-20b") -> str:
    """Send the question plus retrieved context to the LLM and return a grounded answer."""
    client = get_client()
    context = build_context(retrieved_chunks)

    user_message = (
        f"Context from the document:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content
