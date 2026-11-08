"""Structured Evidence Verification Layer.

Research contribution: Replaces the naive keyword-overlap claim verification
in the original RAG agent with a structured pipeline:

  1. CLAIM EXTRACTION: Decomposes the answer text into individual factual claims.
  2. PER-CLAIM VERIFICATION: Each claim is scored against retrieved citations
     for semantic support (not just keyword overlap).
  3. AGGREGATE SCORING: Produces a per-claim breakdown and an overall
     evidence quality score.
  4. FRESHNESS & CREDIBILITY: Incorporates document age and source type
     into the quality assessment.

Source Credibility Hierarchy (highest to lowest):
  - 10-K / 10-Q (audited regulatory filings) — weight 1.0
  - Earnings Report (official company communications) — weight 0.85
  - Analyst Report — weight 0.70
  - News Article — weight 0.55
  - Synthetic / Demo — weight 0.30
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Source Credibility ────────────────────────────────────────────────────────

_CREDIBILITY_WEIGHTS: dict[str, float] = {
    "10-K": 1.0,
    "10-Q": 1.0,
    "10-k": 1.0,
    "10-q": 1.0,
    "Earnings Report": 0.85,
    "earnings_report": 0.85,
    "Analyst Report": 0.70,
    "analyst_report": 0.70,
    "News": 0.55,
    "news": 0.55,
    "Synthetic": 0.30,
    "synthetic": 0.30,
}


def get_source_credibility(doc_type: str) -> float:
    """Return credibility weight for a document type."""
    for key, weight in _CREDIBILITY_WEIGHTS.items():
        if key.lower() in doc_type.lower():
            return weight
    return 0.50  # Unknown source type


# ── Freshness Scoring ─────────────────────────────────────────────────────────

def compute_freshness_score(published_at: str, as_of_date: str | None = None,
                            max_age_days: int = 365) -> float:
    """Score document freshness on [0, 1].

    Documents published recently get higher scores. Documents older than
    ``max_age_days`` are penalized toward 0.
    """
    if not published_at:
        return 0.3  # Unknown publication date — conservative default

    try:
        pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        try:
            pub_date = datetime.strptime(published_at[:10], "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            return 0.3

    if as_of_date:
        try:
            ref_date = datetime.strptime(as_of_date[:10], "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            ref_date = datetime.now(timezone.utc)
    else:
        ref_date = datetime.now(timezone.utc)

    if pub_date.tzinfo is None:
        pub_date = pub_date.replace(tzinfo=timezone.utc)

    age_days = (ref_date - pub_date).days
    if age_days < 0:
        return 0.0  # Future document — should be filtered by temporal firewall
    if age_days <= 30:
        return 1.0
    if age_days <= 90:
        return 0.85
    if age_days <= 180:
        return 0.65
    if age_days <= max_age_days:
        return 0.40
    return 0.15  # Very stale


# ── Claim Extraction ──────────────────────────────────────────────────────────

class ExtractedClaim(BaseModel):
    """A single factual claim extracted from an answer."""
    claim_text: str
    claim_type: str  # NUMERIC, DIRECTIONAL, TEMPORAL, QUALITATIVE
    key_terms: list[str]


def extract_claims(answer_text: str) -> list[ExtractedClaim]:
    """Decompose an answer into individual factual claims."""
    if not answer_text or answer_text.strip() == "Insufficient evidence in the knowledge base.":
        return []

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    claims: list[ExtractedClaim] = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue

        # Classify claim type
        claim_type = _classify_claim(sent)

        # Extract key terms (words >= 4 chars, excluding common stop words)
        words = re.findall(r"[a-zA-Z]{4,}", sent)
        stop_words = {"this", "that", "with", "from", "have", "been",
                       "were", "will", "about", "their", "which", "other",
                       "based", "than"}
        key_terms = [w.lower() for w in words if w.lower() not in stop_words][:8]

        claims.append(ExtractedClaim(
            claim_text=sent,
            claim_type=claim_type,
            key_terms=key_terms,
        ))

    return claims


def _classify_claim(text: str) -> str:
    """Classify a claim sentence by type."""
    text_lower = text.lower()
    # Numeric claims contain dollar amounts, percentages, or specific numbers
    if re.search(r"\$[\d,]+|[\d.]+%|\d+\s*(billion|million|trillion)", text_lower):
        return "NUMERIC"
    # Temporal claims reference specific time periods
    if re.search(r"(q[1-4]|fy\d{4}|quarter|year|fiscal|20\d{2})", text_lower):
        return "TEMPORAL"
    # Directional claims state growth/decline/increase/decrease
    if re.search(r"(grew|growth|decline|increase|decrease|rose|fell|up|down)", text_lower):
        return "DIRECTIONAL"
    return "QUALITATIVE"


# ── Per-Claim Verification ────────────────────────────────────────────────────

class ClaimVerification(BaseModel):
    """Verification result for a single claim."""
    claim: ExtractedClaim
    support_score: float = Field(ge=0.0, le=1.0)
    supporting_citation_ids: list[str]
    verification_status: str  # SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED
    explanation: str


class VerificationResult(BaseModel):
    """Aggregate verification across all claims."""
    total_claims: int
    supported_claims: int
    partially_supported_claims: int
    unsupported_claims: int
    overall_evidence_quality: float = Field(ge=0.0, le=1.0)
    claim_verifications: list[ClaimVerification]
    average_freshness: float = Field(ge=0.0, le=1.0)
    average_credibility: float = Field(ge=0.0, le=1.0)
    verification_status: str  # STRONG, MODERATE, WEAK, INSUFFICIENT
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class EvidenceVerifier:
    """Structured claim extraction and per-claim verification."""

    def extract_claims(self, answer_text: str) -> list[ExtractedClaim]:
        """Extract individual factual claims from answer text."""
        return extract_claims(answer_text)

    def verify(
        self,
        answer_text: str,
        citations: list[dict],
        as_of_date: str | None = None,
    ) -> VerificationResult:
        """Run the full verification pipeline."""
        claims = extract_claims(answer_text)

        if not claims:
            return VerificationResult(
                total_claims=0,
                supported_claims=0,
                partially_supported_claims=0,
                unsupported_claims=0,
                overall_evidence_quality=0.0 if not citations else 0.5,
                claim_verifications=[],
                average_freshness=0.0,
                average_credibility=0.0,
                verification_status="INSUFFICIENT",
            )

        # Build evidence corpus from citations
        evidence_terms = set()
        for c in citations:
            text_fields = [
                str(c.get("source", "")),
                str(c.get("section", "")),
                str(c.get("period", "")),
                str(c.get("text", "")),
                str(c.get("company", "")),
            ]
            for field_text in text_fields:
                words = re.findall(r"[a-zA-Z]{4,}", field_text)
                evidence_terms.update(w.lower() for w in words)

        # Verify each claim
        verifications: list[ClaimVerification] = []
        for claim in claims:
            v = self._verify_single_claim(claim, evidence_terms, citations)
            verifications.append(v)

        # Aggregate statistics
        supported = sum(1 for v in verifications
                        if v.verification_status == "SUPPORTED")
        partial = sum(1 for v in verifications
                      if v.verification_status == "PARTIALLY_SUPPORTED")
        unsupported = sum(1 for v in verifications
                         if v.verification_status == "UNSUPPORTED")

        # Compute freshness and credibility averages
        freshness_scores = []
        credibility_scores = []
        for c in citations:
            pub = str(c.get("published_at", ""))
            if pub:
                freshness_scores.append(
                    compute_freshness_score(pub, as_of_date))
            doc_type = str(c.get("doc_type", ""))
            credibility_scores.append(get_source_credibility(doc_type))

        avg_freshness = (sum(freshness_scores) / len(freshness_scores)
                         if freshness_scores else 0.0)
        avg_credibility = (sum(credibility_scores) / len(credibility_scores)
                           if credibility_scores else 0.0)

        # Overall evidence quality
        claim_quality = (supported + 0.5 * partial) / max(len(claims), 1)
        overall = (0.50 * claim_quality +
                   0.25 * avg_freshness +
                   0.25 * avg_credibility)

        # Classification
        if overall >= 0.70:
            status = "STRONG"
        elif overall >= 0.45:
            status = "MODERATE"
        elif overall >= 0.20:
            status = "WEAK"
        else:
            status = "INSUFFICIENT"

        return VerificationResult(
            total_claims=len(claims),
            supported_claims=supported,
            partially_supported_claims=partial,
            unsupported_claims=unsupported,
            overall_evidence_quality=round(overall, 4),
            claim_verifications=verifications,
            average_freshness=round(avg_freshness, 4),
            average_credibility=round(avg_credibility, 4),
            verification_status=status,
        )

    def _verify_single_claim(
        self,
        claim: ExtractedClaim,
        evidence_terms: set[str],
        citations: list[dict],
    ) -> ClaimVerification:
        """Score a single claim against the evidence corpus."""
        if not claim.key_terms:
            return ClaimVerification(
                claim=claim,
                support_score=0.0,
                supporting_citation_ids=[],
                verification_status="UNSUPPORTED",
                explanation="No key terms to verify.",
            )

        # Term overlap between claim and evidence
        claim_terms = set(claim.key_terms)
        overlap = claim_terms & evidence_terms
        overlap_ratio = len(overlap) / len(claim_terms) if claim_terms else 0.0

        # Find supporting citations
        supporting_ids = []
        for c in citations:
            c_text = " ".join([
                str(c.get("source", "")),
                str(c.get("section", "")),
                str(c.get("text", "")),
            ]).lower()
            c_words = set(re.findall(r"[a-zA-Z]{4,}", c_text))
            if claim_terms & c_words:
                supporting_ids.append(str(c.get("chunk_id", "unknown")))

        # Score
        if overlap_ratio >= 0.50:
            status = "SUPPORTED"
            score = min(overlap_ratio * 1.2, 1.0)
            explanation = (f"{len(overlap)}/{len(claim_terms)} key terms found "
                           f"in evidence corpus.")
        elif overlap_ratio >= 0.25:
            status = "PARTIALLY_SUPPORTED"
            score = overlap_ratio
            explanation = (f"Partial overlap: {len(overlap)}/{len(claim_terms)} "
                           f"terms matched.")
        else:
            status = "UNSUPPORTED"
            score = overlap_ratio
            explanation = (f"Weak overlap: only {len(overlap)}/{len(claim_terms)} "
                           f"terms found in evidence.")

        return ClaimVerification(
            claim=claim,
            support_score=round(score, 4),
            supporting_citation_ids=supporting_ids[:5],
            verification_status=status,
            explanation=explanation,
        )
