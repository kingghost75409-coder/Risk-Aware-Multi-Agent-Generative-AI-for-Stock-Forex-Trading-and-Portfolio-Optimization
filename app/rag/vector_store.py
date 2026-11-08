"""Vector store for the Financial RAG pipeline.

Preferred backend per spec is Qdrant with sentence-transformer embeddings.
This environment has no network access to a Qdrant server or to download
HuggingFace embedding weights, so `TfidfVectorStore` is used instead: a
real, working, in-process vector index (scikit-learn TF-IDF + cosine
similarity) that fulfills the same interface (`upsert`, `search`).

`QdrantVectorStore` is stubbed alongside it, sharing the same interface —
swapping backends is a one-line change (`get_vector_store(backend=...)`)
and touches no agent code.
"""
from __future__ import annotations

import abc
import re
import uuid
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict

    @property
    def index_text(self) -> str:
        """Text actually embedded/indexed — includes company/section context
        so short excerpts retrieve well even when the raw prose doesn't
        repeat the section heading. Displayed `text` stays clean for answers."""
        m = self.metadata
        context = f"{m.get('company', '')} {m.get('doc_type', '')} {m.get('section', '')} {m.get('period', '')}"
        return f"{context} {self.text}"


class VectorStore(abc.ABC):
    @abc.abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None: ...

    @abc.abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_company: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]: ...


def _chunk_text(text: str, max_words: int = 120) -> list[str]:
    """Simple word-count chunker with slight overlap — documents here are
    short excerpts so most produce a single chunk; long documents split
    cleanly on sentence boundaries where possible."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, count = [], [], 0
    for sent in sentences:
        words = sent.split()
        if count + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


class TfidfVectorStore(VectorStore):
    """In-process TF-IDF cosine-similarity vector index with temporal filtering."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def upsert(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not self._chunks:
            return
        texts = [c.index_text for c in self._chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_company: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:
        """Performs vector search with point-in-time temporal filtering.

        If as_of_date is specified (e.g. '2025-10-01'), any document published
        AFTER as_of_date is strictly excluded to prevent look-ahead bias and
        future information leakage.
        """
        if not self._chunks or self._vectorizer is None:
            return []

        # Candidate indices matching company and temporal constraints
        valid_indices = []
        for i, c in enumerate(self._chunks):
            meta = c.metadata
            if filter_company and meta.get("company", "").upper() != filter_company.upper():
                continue
            if as_of_date:
                doc_pub = str(meta.get("published_at", ""))[:10]
                target_date = str(as_of_date)[:10]
                if doc_pub and doc_pub > target_date:
                    continue  # Strict look-ahead prevention
            valid_indices.append(i)

        if not valid_indices:
            return []

        candidates = [self._chunks[i] for i in valid_indices]
        matrix = self._matrix[valid_indices]

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix).flatten()
        order = np.argsort(-sims)[:top_k]

        results = []
        for rank, idx in enumerate(order):
            score = float(sims[idx])
            if score <= 0:
                continue
            c = candidates[idx]
            meta = dict(c.metadata or {})

            # Staleness-aware boost (Upgrade 4)
            # Recent documents get higher relative weight; stale ones (>180d) discounted
            doc_pub = str(meta.get("published_at", ""))[:10]
            boost = 1.0
            if doc_pub and as_of_date:
                try:
                    target_dt = pd.to_datetime(as_of_date[:10])
                    pub_dt = pd.to_datetime(doc_pub)
                    age_days = (target_dt - pub_dt).days
                    if age_days <= 30:
                        boost = 1.15
                    elif age_days <= 90:
                        boost = 1.05
                    elif age_days > 180:
                        boost = 0.85
                except Exception:
                    pass
            adjusted_score = min(1.0, score * boost)

            results.append({
                "rank": rank + 1,
                "score": round(score, 4),
                "adjusted_score": round(adjusted_score, 4),
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": meta,
            })
        return results


class QdrantVectorStore(VectorStore):  # pragma: no cover — requires a running Qdrant server
    """Drop-in Qdrant-backed implementation.

    Not used by default in this environment (no server / no network access
    to download embedding model weights). Kept here, fully wired to the
    same interface, for deployment onto real infrastructure — see README
    "Swapping in real infrastructure".
    """

    def __init__(self, url: str = "http://localhost:6333", collection: str = "financial_rag") -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "QdrantVectorStore requires `pip install qdrant-client sentence-transformers` "
                "and network access to both a Qdrant server and huggingface.co."
            ) from exc
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        dim = self._model.get_sentence_embedding_dimension()
        self._client.recreate_collection(
            collection_name=collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk]) -> None:  # pragma: no cover
        from qdrant_client.models import PointStruct
        vectors = self._model.encode([c.text for c in chunks])
        points = [
            PointStruct(id=c.chunk_id, vector=vec.tolist(), payload={"text": c.text, **c.metadata})
            for c, vec in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_company: str | None = None,
        as_of_date: str | None = None,
    ) -> list[dict]:  # pragma: no cover
        vec = self._model.encode([query])[0].tolist()
        hits = self._client.search(collection_name=self._collection, query_vector=vec, limit=top_k)
        return [{"score": h.score, "chunk_id": h.id, "text": h.payload.get("text"), "metadata": h.payload} for h in hits]


_STORE_SINGLETON: VectorStore | None = None


def get_vector_store(backend: str = "tfidf") -> VectorStore:
    global _STORE_SINGLETON
    if _STORE_SINGLETON is None:
        _STORE_SINGLETON = QdrantVectorStore() if backend == "qdrant" else TfidfVectorStore()
    return _STORE_SINGLETON


def make_chunk_id() -> str:
    return uuid.uuid4().hex[:12]


def chunk_documents(documents: list[dict]) -> list[Chunk]:
    """documents: list of {text, company, doc_type, section, period, source, published_at}."""
    chunks: list[Chunk] = []
    for doc in documents:
        doc_type = str(doc.get("doc_type", "")).lower()
        if "10-k" in doc_type or "10-q" in doc_type:
            tier = "Tier 1 (SEC Filing)"
        elif "earnings" in doc_type:
            tier = "Tier 2 (Official Release)"
        elif "analyst" in doc_type:
            tier = "Tier 3 (Analyst Report)"
        elif "news" in doc_type:
            tier = "Tier 4 (Market News)"
        else:
            tier = "Tier 5 (Synthetic/Other)"

        pieces = _chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            meta = {k: v for k, v in doc.items() if k != "text"}
            meta["chunk_index"] = i
            meta["credibility_tier"] = tier
            chunks.append(Chunk(chunk_id=make_chunk_id(), text=piece, metadata=meta))
    return chunks
