"""Temporal Financial Evidence Firewall.

Research contribution: ALL data-dependent decisions at decision-time T must use
ONLY information that was genuinely available at T.  This module provides a
centralized, auditable enforcement layer that:

  1. Truncates OHLCV DataFrames to ``as_of_date`` (no future price bars).
  2. Validates RAG/evidence document ``published_at`` dates.
  3. Ensures structured facts respect ``available_timestamp``.
  4. Logs every truncation/violation for the audit trail.

SAFETY RULE: The firewall can only REMOVE data (constrict), never add data
or shift boundaries forward.  If ``as_of_date`` is ``None`` the firewall
is a no-op (live/latest mode).
"""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field

import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FirewallAuditEntry:
    """Single audit record produced by a firewall validation call."""
    data_type: str          # "ohlcv" | "rag_document" | "structured_fact"
    symbol: str
    as_of_date: str
    original_count: int
    accepted_count: int
    rejected_count: int
    violations: list[str] = field(default_factory=list)

    @property
    def had_violations(self) -> bool:
        return self.rejected_count > 0


@dataclass
class TemporalFirewallReport:
    """Aggregate report for all firewall checks in a single analysis cycle."""
    as_of_date: str | None
    entries: list[FirewallAuditEntry] = field(default_factory=list)
    total_violations: int = 0

    def add(self, entry: FirewallAuditEntry) -> None:
        self.entries.append(entry)
        self.total_violations += entry.rejected_count

    @property
    def is_clean(self) -> bool:
        return self.total_violations == 0

    def summary(self) -> str:
        if self.as_of_date is None:
            return "Temporal firewall: LIVE mode (no as_of_date constraint)."
        parts = [f"Temporal firewall as_of={self.as_of_date}: "]
        if self.is_clean:
            parts.append("CLEAN — no future data detected.")
        else:
            parts.append(f"{self.total_violations} future data point(s) removed.")
            for e in self.entries:
                if e.had_violations:
                    parts.append(f"  [{e.data_type}] {e.symbol}: "
                                 f"rejected {e.rejected_count}/{e.original_count}")
        return " ".join(parts)


class TemporalFirewall:
    """Centralized point-in-time data boundary enforcement."""

    def __init__(self, as_of_date: str | None = None) -> None:
        """
        Parameters
        ----------
        as_of_date : str | None
            ISO date string (``YYYY-MM-DD``) representing the decision point.
            ``None`` means live/latest mode — no truncation is performed.
        """
        self.as_of_date = as_of_date
        self.report = TemporalFirewallReport(as_of_date=as_of_date)

    # ── OHLCV DataFrame enforcement ─────────────────────────────────────

    def enforce_ohlcv(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Truncate a price DataFrame so no bar has a timestamp after ``as_of_date``.

        Returns a (possibly shorter) DataFrame.  The original is never mutated.
        """
        if self.as_of_date is None or df.empty:
            return df

        boundary = pd.Timestamp(self.as_of_date, tz="UTC")
        original_count = len(df)

        # Ensure index is tz-aware UTC for comparison
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        elif str(idx.tz) != "UTC":
            idx = idx.tz_convert("UTC")

        mask = idx <= boundary
        truncated = df.loc[mask]
        rejected = original_count - len(truncated)

        violations = []
        if rejected > 0:
            violations.append(
                f"Removed {rejected} future bar(s) after {self.as_of_date} "
                f"for {symbol}"
            )
            logger.warning("Temporal firewall: %s", violations[0])

        self.report.add(FirewallAuditEntry(
            data_type="ohlcv",
            symbol=symbol,
            as_of_date=self.as_of_date,
            original_count=original_count,
            accepted_count=len(truncated),
            rejected_count=rejected,
            violations=violations,
        ))
        return truncated

    # ── RAG / evidence document enforcement ──────────────────────────────

    def filter_documents(self, documents: list[dict], symbol: str) -> list[dict]:
        """Remove documents whose ``published_at`` is after ``as_of_date``."""
        if self.as_of_date is None:
            return documents

        target = self.as_of_date[:10]
        original_count = len(documents)
        accepted = []
        violations = []

        for doc in documents:
            pub = str(doc.get("published_at", ""))[:10]
            if pub and pub > target:
                violations.append(
                    f"Rejected doc '{doc.get('source', '?')}' "
                    f"published {pub} > as_of {target}"
                )
            else:
                accepted.append(doc)

        if violations:
            logger.warning(
                "Temporal firewall: filtered %d/%d documents for %s",
                len(violations), original_count, symbol,
            )

        self.report.add(FirewallAuditEntry(
            data_type="rag_document",
            symbol=symbol,
            as_of_date=self.as_of_date,
            original_count=original_count,
            accepted_count=len(accepted),
            rejected_count=len(violations),
            violations=violations,
        ))
        return accepted

    # ── Structured facts enforcement ─────────────────────────────────────

    def filter_structured_facts(self, facts: list[dict], symbol: str) -> list[dict]:
        """Remove structured facts not yet available at ``as_of_date``."""
        if self.as_of_date is None:
            return facts

        target = self.as_of_date[:10]
        original_count = len(facts)
        accepted = []
        violations = []

        for fact in facts:
            available = str(fact.get("available_timestamp", fact.get("period", "")))[:10]
            # If available_timestamp is set, use it; otherwise be conservative
            if available and available > target:
                violations.append(
                    f"Rejected fact '{fact.get('metric', '?')}' "
                    f"available {available} > as_of {target}"
                )
            else:
                accepted.append(fact)

        self.report.add(FirewallAuditEntry(
            data_type="structured_fact",
            symbol=symbol,
            as_of_date=self.as_of_date,
            original_count=original_count,
            accepted_count=len(accepted),
            rejected_count=len(violations),
            violations=violations,
        ))
        return accepted

    # ── Convenience ──────────────────────────────────────────────────────

    def get_audit_summary(self) -> str:
        return self.report.summary()
