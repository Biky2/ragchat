"""Full RAG chain: retrieve → prompt → LLM → parse citations."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from config.settings import get_settings
from core.llm import get_llm_response, stream_llm_response
from generation.citation_parser import parse_citations
from generation.prompt_builder import FALLBACK_ANSWER, SYSTEM_PROMPT, build_prompt
from memory.postgres_client import PostgresClient, get_postgres
from memory.session_store import SessionStore, get_session_store
from retrieval.hybrid import HybridRetriever, HybridResult


@dataclass
class ChainResult:
    answer: str
    citations: list[dict[str, Any]]
    chunks_used: int
    retrieval_scores: list[dict[str, Any]]
    latency_ms: int
    session_id: str
    chat_history_id: str | None = None


class RAGChain:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        postgres: PostgresClient | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.settings = get_settings()
        self.retriever = retriever or HybridRetriever()
        self._postgres = postgres
        self._session_store = session_store

    async def _get_postgres(self) -> PostgresClient:
        if self._postgres is None:
            self._postgres = await get_postgres()
        return self._postgres

    async def _get_session_store(self) -> SessionStore:
        if self._session_store is None:
            self._session_store = get_session_store()
        return self._session_store

    async def _build_conversation_context(self, session_id: str) -> str:
        db = await self._get_postgres()
        store = await self._get_session_store()

        history = store.get_recent_exchanges(
            session_id, self.settings.context_exchanges
        )
        if not history:
            history = await db.get_recent_history(
                session_id, self.settings.context_exchanges
            )
            for item in history:
                store.add_exchange(session_id, item["query"], item["answer"])

        lines: list[str] = []
        for item in history:
            lines.append(f"User: {item['query']}")
            lines.append(f"Assistant: {item['answer']}")
        return "\n".join(lines)

    def _build_retrieval_scores(self, chunks: list[HybridResult]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": c.id,
                "source_file": c.metadata.get("source_file"),
                "dense_score": c.dense_score,
                "sparse_score": c.sparse_score,
                "rrf_score": c.rrf_score,
            }
            for c in chunks
        ]

    async def run(
        self,
        query: str,
        session_id: str,
        top_k: int | None = None,
    ) -> ChainResult:
        start = time.perf_counter()
        top_k = top_k or self.settings.default_top_k

        conversation_context = await self._build_conversation_context(session_id)
        enriched_query = query
        if conversation_context:
            enriched_query = f"{conversation_context}\n\nCurrent question: {query}"

        chunks = await self.retriever.retrieve(enriched_query, top_k=top_k)

        if not chunks:
            answer = FALLBACK_ANSWER
            citations: list[dict[str, Any]] = []
            retrieval_scores: list[dict[str, Any]] = []
        else:
            system_prompt, user_prompt = build_prompt(
                query, chunks, conversation_context
            )
            answer = await get_llm_response(user_prompt, system_prompt)
            citations = parse_citations(answer, chunks)
            retrieval_scores = self._build_retrieval_scores(chunks)

        latency_ms = int((time.perf_counter() - start) * 1000)

        db = await self._get_postgres()
        store = await self._get_session_store()

        chat_history_id = await db.save_chat_history(
            session_id=session_id,
            query=query,
            answer=answer,
            citations=citations,
            retrieval_scores=retrieval_scores,
            chunks_used=len(chunks),
            latency_ms=latency_ms,
        )

        store.add_exchange(session_id, query, answer)
        store.trim_session(session_id, self.settings.max_session_exchanges)

        return ChainResult(
            answer=answer,
            citations=citations,
            chunks_used=len(chunks),
            retrieval_scores=retrieval_scores,
            latency_ms=latency_ms,
            session_id=session_id,
            chat_history_id=chat_history_id,
        )

    async def stream(
        self,
        query: str,
        session_id: str,
        top_k: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        start = time.perf_counter()
        top_k = top_k or self.settings.default_top_k

        conversation_context = await self._build_conversation_context(session_id)
        enriched_query = query
        if conversation_context:
            enriched_query = f"{conversation_context}\n\nCurrent question: {query}"

        chunks = await self.retriever.retrieve(enriched_query, top_k=top_k)

        if not chunks:
            yield {"token": FALLBACK_ANSWER, "done": False}
            latency_ms = int((time.perf_counter() - start) * 1000)
            db = await self._get_postgres()
            store = await self._get_session_store()
            chat_history_id = await db.save_chat_history(
                session_id=session_id,
                query=query,
                answer=FALLBACK_ANSWER,
                citations=[],
                retrieval_scores=[],
                chunks_used=0,
                latency_ms=latency_ms,
            )
            store.add_exchange(session_id, query, FALLBACK_ANSWER)
            yield {
                "token": "",
                "done": True,
                "citations": [],
                "retrieval_scores": [],
                "chunks_used": 0,
                "latency_ms": latency_ms,
                "chat_history_id": chat_history_id,
            }
            return

        system_prompt, user_prompt = build_prompt(query, chunks, conversation_context)
        full_answer = ""

        async for token in stream_llm_response(user_prompt, system_prompt):
            full_answer += token
            yield {"token": token, "done": False}

        citations = parse_citations(full_answer, chunks)
        retrieval_scores = self._build_retrieval_scores(chunks)
        latency_ms = int((time.perf_counter() - start) * 1000)

        db = await self._get_postgres()
        store = await self._get_session_store()
        chat_history_id = await db.save_chat_history(
            session_id=session_id,
            query=query,
            answer=full_answer,
            citations=citations,
            retrieval_scores=retrieval_scores,
            chunks_used=len(chunks),
            latency_ms=latency_ms,
        )
        store.add_exchange(session_id, query, full_answer)
        store.trim_session(session_id, self.settings.max_session_exchanges)

        yield {
            "token": "",
            "done": True,
            "citations": citations,
            "retrieval_scores": retrieval_scores,
            "chunks_used": len(chunks),
            "latency_ms": latency_ms,
            "chat_history_id": chat_history_id,
        }
