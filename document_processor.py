"""Load a PDF and split it into overlapping text chunks for embedding."""

from typing import List, Dict
from pypdf import PdfReader


def extract_text_by_page(pdf_path: str) -> List[Dict]:
    """Extract text from every page of a PDF.

    Returns a list like [{"page": 1, "text": "..."}, ...]
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages: List[Dict], chunk_size: int = 350, overlap: int = 60) -> List[Dict]:
    """Split page text into overlapping word-based chunks.

    Each chunk remembers which page it came from, which is how DocuChat
    is able to cite sources later.
    """
    chunks = []
    chunk_id = 0
    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            if chunk_text.strip():
                chunks.append({
                    "page": page["page"],
                    "chunk_id": chunk_id,
                    "text": chunk_text
                })
                chunk_id += 1
            start += chunk_size - overlap
    return chunks
