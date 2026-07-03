"""In-memory session store for active chat conversations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Exchange:
    query: str
    answer: str


@dataclass
class Session:
    session_id: str
    exchanges: list[Exchange] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = defaultdict(lambda: Session(""))
        self._lock = Lock()

    def add_exchange(self, session_id: str, query: str, answer: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
            self._sessions[session_id].exchanges.append(Exchange(query=query, answer=answer))

    def get_recent_exchanges(self, session_id: str, limit: int = 3) -> list[dict[str, str]]:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.exchanges:
                return []
            recent = session.exchanges[-limit:]
            return [{"query": e.query, "answer": e.answer} for e in recent]

    def trim_session(self, session_id: str, max_exchanges: int = 6) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session and len(session.exchanges) > max_exchanges:
                session.exchanges = session.exchanges[-max_exchanges:]

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def get_exchange_count(self, session_id: str) -> int:
        with self._lock:
            session = self._sessions.get(session_id)
            return len(session.exchanges) if session else 0


_session_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
