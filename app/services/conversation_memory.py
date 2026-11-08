"""Conversation memory for the trading chatbot.

Stores chat history per session with context window management.
Uses in-memory storage with optional SQLite persistence.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool_used: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """Manages chat history with a sliding context window."""

    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self._history: deque[ChatMessage] = deque(maxlen=max_messages)
        self.user_prefs: dict[str, Any] = {
            "experience_level": "intermediate",  # beginner, intermediate, advanced
            "favorite_symbols": [],
            "risk_tolerance": "moderate",  # conservative, moderate, aggressive
        }

    def add_message(self, role: str, content: str, tool_used: str | None = None,
                    metadata: dict | None = None) -> None:
        self._history.append(ChatMessage(
            role=role, content=content, tool_used=tool_used,
            metadata=metadata or {},
        ))

    def get_context(self, last_n: int = 10) -> list[dict[str, str]]:
        """Return last N messages formatted for LLM context."""
        messages = list(self._history)[-last_n:]
        return [{"role": m.role, "content": m.content} for m in messages]

    def get_history(self, last_n: int = 10) -> list[dict[str, str]]:
        """Alias for get_context for LLM generation compatibility."""
        return self.get_context(last_n=last_n)

    def get_context_string(self, last_n: int = 6) -> str:
        """Return recent chat history as a formatted string."""
        messages = list(self._history)[-last_n:]
        if not messages:
            return ""
        lines = []
        for m in messages:
            prefix = "User" if m.role == "user" else "Assistant"
            lines.append(f"{prefix}: {m.content[:200]}")
        return "\n".join(lines)

    @property
    def message_count(self) -> int:
        return len(self._history)

    def clear(self) -> None:
        self._history.clear()

    def update_preference(self, key: str, value: Any) -> None:
        self.user_prefs[key] = value


# Singleton per-session memory
_memories: dict[str, ConversationMemory] = {}


def get_memory(session_id: str = "default") -> ConversationMemory:
    """Get or create conversation memory for a session."""
    if session_id not in _memories:
        _memories[session_id] = ConversationMemory()
    return _memories[session_id]
