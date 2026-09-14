"""FAISS-backed vector store for semantic search over document chunks.

Embeddings are generated locally with fastembed (ONNX Runtime under the
hood). This avoids the torch + scikit-learn dependency chain that can
trigger blocked-DLL errors on locked-down Windows machines, and it also
avoids external API rate limits since nothing is called over the network
after the model is downloaded once.
"""

from typing import List, Dict
import numpy as np
import faiss
from fastembed import TextEmbedding


class VectorStore:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)
        self.index = None
        self.chunks: List[Dict] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = list(self.model.embed(texts))
        return np.array(vectors, dtype="float32")

    def build(self, chunks: List[Dict]) -> None:
        """Embed all chunks and build a FAISS index."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)
        faiss.normalize_L2(embeddings)  # so inner product == cosine similarity

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        """Return the top_k most relevant chunks for a query, with similarity scores."""
        if self.index is None:
            raise ValueError("Vector store is empty. Call build() first.")

        query_embedding = self._embed([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "page": chunk["page"],
                "score": float(score)
            })
        return results
