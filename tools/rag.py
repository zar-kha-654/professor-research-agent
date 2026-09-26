"""
Lightweight, session-scoped RAG pipeline.

Pipeline: clean text -> chunk -> embed -> FAISS index -> retrieve top-k
relevant chunks for a given professor/field query.

Nothing here is persisted to disk: a new SessionVectorStore is created per
research session and discarded when the Streamlit session ends, per the
"do not permanently store university data" requirement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from utils.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME, RAG_TOP_K
from utils.helpers import chunk_text

_model = None  # lazy-loaded singleton


def _get_embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


@dataclass
class DocChunk:
    text: str
    source_url: str
    source_title: str = ""


@dataclass
class SessionVectorStore:
    """In-memory FAISS index scoped to a single research session."""

    chunks: list[DocChunk] = field(default_factory=list)
    _index = None
    _dim: int = 0

    def add_page(self, text: str, source_url: str, source_title: str = "") -> int:
        """Chunk a page's clean text and add it to the index. Returns count added."""
        pieces = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not pieces:
            return 0
        embedder = _get_embedder()
        vectors = embedder.encode(pieces, show_progress_bar=False, normalize_embeddings=True)
        vectors = np.asarray(vectors, dtype="float32")

        import faiss  # imported lazily so app can start without it if unused

        if self._index is None:
            self._dim = vectors.shape[1]
            self._index = faiss.IndexFlatIP(self._dim)

        self._index.add(vectors)
        for p in pieces:
            self.chunks.append(DocChunk(text=p, source_url=source_url, source_title=source_title))
        return len(pieces)

    def is_empty(self) -> bool:
        return self._index is None or self._index.ntotal == 0

    def query(self, query_text: str, top_k: int = RAG_TOP_K) -> list[DocChunk]:
        """Return the top-k most relevant chunks for a query."""
        if self.is_empty():
            return []
        embedder = _get_embedder()
        qvec = embedder.encode([query_text], show_progress_bar=False, normalize_embeddings=True)
        qvec = np.asarray(qvec, dtype="float32")
        k = min(top_k, self._index.ntotal)
        _scores, idxs = self._index.search(qvec, k)
        return [self.chunks[i] for i in idxs[0] if 0 <= i < len(self.chunks)]

    def context_for(self, professor_name: str, fields: list[str], max_chars: int = 6000) -> tuple[str, str]:
        """Build a retrieval query from the professor name + requested fields,
        and return (concatenated_context_text, best_source_url)."""
        query = f"{professor_name} " + " ".join(fields)
        results = self.query(query)
        if not results:
            return "", ""
        combined = []
        total = 0
        for c in results:
            if total >= max_chars:
                break
            combined.append(c.text)
            total += len(c.text)
        best_source = results[0].source_url
        return "\n\n".join(combined), best_source
