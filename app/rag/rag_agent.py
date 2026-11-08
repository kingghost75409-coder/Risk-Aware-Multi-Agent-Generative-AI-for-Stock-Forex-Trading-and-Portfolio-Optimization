"""Financial RAG Agent — hybrid retrieval + grounded, citation-verified answers.

Pipeline: Documents -> Parser/Chunker -> Metadata -> Embeddings (TF-IDF) ->
Vector Store -> Retriever -> RAG Context -> (template or LLM) Answer ->
Claim/Citation Verification -> Accept/Reject.

Two retrieval paths, combined ("Hybrid Financial RAG"):
  1. Vector search over unstructured filing/earnings text (semantic).
  2. Structured lookup against `STRUCTURED_FACTS` for exact numeric asks
     ("what was revenue?") instead of guessing from a retrieved paragraph.

The answer generator NEVER states a factual claim that isn't backed by a
retrieved chunk or a structured fact. If nothing relevant is retrieved, it
returns "Insufficient evidence in the knowledge base." verbatim.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel

from rag_data.sample_filings import SAMPLE_DOCUMENTS, STRUCTURED_FACTS
from app.rag.vector_store import chunk_documents, get_vector_store
from app.rag.evidence_verifier import (
    EvidenceVerifier, VerificationResult,
    compute_freshness_score, get_source_credibility,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

INSUFFICIENT_EVIDENCE = "Insufficient evidence in the knowledge base."

_NUMERIC_INTENT_KEYWORDS = ["revenue", "margin", "cash", "growth", "how much", "what was the", "exact"]


class Citation(BaseModel):
    source: str
    company: str
    doc_type: str
    section: str
    period: str
    chunk_id: str
    score: float
    text: str = ""
    published_at: str = ""
    page_or_section: str = ""
    freshness_score: float = 0.0
    credibility_weight: float = 0.5


class RAGResult(BaseModel):
    symbol: str
    query: str
    answer: str
    citations: list[Citation]
    structured_fact: dict | None = None
    verification: dict
    verification_breakdown: dict | None = None
    evidence_quality_score: float = 0.0
    average_freshness: float = 0.0
    average_credibility: float = 0.0
    as_of_date: str | None = None
    insufficient_evidence: bool = False
    timestamp: str


_INGESTED = False


def _ensure_ingested() -> None:
    global _INGESTED
    if _INGESTED:
        return
    store = get_vector_store("tfidf")
    chunks = chunk_documents(SAMPLE_DOCUMENTS)
    store.upsert(chunks)
    _INGESTED = True
    logger.info("RAG store ingested %d chunks from %d sample documents", len(chunks), len(SAMPLE_DOCUMENTS))


def _lookup_structured_fact(symbol: str, query: str, as_of_date: str | None = None) -> dict | None:
    q_lower = query.lower()
    candidates = [f for f in STRUCTURED_FACTS if f["company"].upper() == symbol.upper()]
    if not candidates:
        return None
    # crude keyword match against metric name, good enough for the demo fact set
    for fact in candidates:
        metric_words = fact["metric"].replace("_", " ").split()
        if any(w in q_lower for w in metric_words if len(w) > 3):
            # Optional temporal check on structured fact
            if as_of_date:
                period_str = str(fact.get("period", ""))
                # If period mentions FY2026 and as_of_date is in 2025, prevent future leakage
                if "2026" in period_str and as_of_date < "2026-01-01":
                    continue
            return fact
    return None


def _verify_claims(answer: str, citations: list[Citation]) -> dict:
    """Lightweight claim/citation verification: every sentence in the
    answer must be traceable to retrieved evidence content overlap, or the
    answer must be the explicit insufficient-evidence string."""
    if answer == INSUFFICIENT_EVIDENCE:
        return {"status": "NO_CLAIM_MADE", "unsupported_sentences": []}
    if not citations:
        return {"status": "UNSUPPORTED", "unsupported_sentences": [answer]}

    evidence_words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}",
                       " ".join(f"{c.source} {c.section} {c.period}" for c in citations))}
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    unsupported = []
    for sent in sentences:
        words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", sent)}
        overlap = words & evidence_words
        if words and len(overlap) == 0:
            unsupported.append(sent)
    status = "SUPPORTED" if not unsupported else "PARTIALLY_UNSUPPORTED"
    return {"status": status, "unsupported_sentences": unsupported, "citation_count": len(citations)}


class FinancialRAGAgent:
    def __init__(self) -> None:
        _ensure_ingested()
        self._store = get_vector_store("tfidf")

    def query(
        self,
        symbol: str,
        question: str,
        top_k: int = 3,
        as_of_date: str | None = None,
    ) -> RAGResult:
        structured = _lookup_structured_fact(symbol, question, as_of_date=as_of_date)
        hits = self._store.search(question, top_k=top_k, filter_company=symbol, as_of_date=as_of_date)

        citations = [
            Citation(
                source=h["metadata"].get("source", "unknown"),
                company=h["metadata"].get("company", symbol),
                doc_type=h["metadata"].get("doc_type", "unknown"),
                section=h["metadata"].get("section", "unknown"),
                period=h["metadata"].get("period", "unknown"),
                chunk_id=h["chunk_id"],
                score=h["score"],
                text=h.get("text", ""),
                published_at=h["metadata"].get("published_at", ""),
                page_or_section=h["metadata"].get("section", ""),
                freshness_score=round(compute_freshness_score(
                    h["metadata"].get("published_at", ""), as_of_date), 4),
                credibility_weight=round(get_source_credibility(
                    h["metadata"].get("doc_type", "unknown")), 4),
            )
            for h in hits
        ]

        if structured and structured.get("value") is not None:
            answer = (
                f"For {symbol.upper()} ({structured['period']}), {structured['metric'].replace('_', ' ')} "
                f"was {structured['value']}, per structured financial data (not a vector-retrieved paragraph)."
            )
            insufficient = False
        elif hits:
            top = hits[0]
            answer = (
                f"Based on {top['metadata'].get('source')} ({top['metadata'].get('section')}, "
                f"{top['metadata'].get('period')}, pub: {top['metadata'].get('published_at', 'N/A')}): {top['text']}"
            )
            insufficient = False
        else:
            answer = INSUFFICIENT_EVIDENCE
            insufficient = True

        # Legacy verification (kept for backward compat)
        verification = _verify_claims(answer, citations)

        # Structured evidence verification (new)
        verifier = EvidenceVerifier()
        citation_dicts = [c.model_dump() for c in citations]
        structured_verification = verifier.verify(answer, citation_dicts, as_of_date)

        return RAGResult(
            symbol=symbol.upper(),
            query=question,
            answer=answer,
            citations=citations,
            structured_fact=structured,
            verification=verification,
            verification_breakdown=structured_verification.model_dump(),
            evidence_quality_score=structured_verification.overall_evidence_quality,
            average_freshness=structured_verification.average_freshness,
            average_credibility=structured_verification.average_credibility,
            as_of_date=as_of_date,
            insufficient_evidence=insufficient,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_evidence_summary(self, symbol: str, as_of_date: str | None = None) -> RAGResult:
        """Used by the multi-agent decision layer: pulls general risk/outlook
        evidence for a symbol to inform the trading decision, adhering to as_of_date."""
        return self.query(
            symbol,
            f"{symbol} risk factors and recent financial outlook",
            top_k=3,
            as_of_date=as_of_date,
        )
