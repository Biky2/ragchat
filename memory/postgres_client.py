"""Async PostgreSQL client with SQLAlchemy."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config.settings import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[str | None] = mapped_column(Text)
    chunk_count: Mapped[int | None] = mapped_column(Integer, default=0)
    strategy: Mapped[str | None] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(server_default=func.now())
    status: Mapped[str] = mapped_column(Text, default="processing")


class ChunkStore(Base):
    __tablename__ = "chunk_store"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[dict | list | None] = mapped_column(JSONB)
    retrieval_scores: Mapped[dict | list | None] = mapped_column(JSONB)
    chunks_used: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (CheckConstraint("rating IN (1, -1)", name="feedback_rating_check"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chat_history_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_history.id"), nullable=False
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class PostgresClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
        )
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pgcrypto"'))
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def health_check(self) -> bool:
        try:
            async with self.session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error("PostgreSQL health check failed: %s", exc)
            return False

    async def create_document(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        strategy: str,
        status: str = "processing",
    ) -> str:
        async with self.session_factory() as session:
            doc = Document(
                id=uuid.UUID(document_id),
                filename=filename,
                file_type=file_type,
                strategy=strategy,
                status=status,
                chunk_count=0,
            )
            session.add(doc)
            await session.commit()
        return document_id

    async def update_document_status(
        self, document_id: str, status: str, chunk_count: int
    ) -> None:
        async with self.session_factory() as session:
            result = await session.execute(
                select(Document).where(Document.id == uuid.UUID(document_id))
            )
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = status
                doc.chunk_count = chunk_count
                await session.commit()

    async def list_documents(self) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(Document).order_by(Document.uploaded_at.desc())
            )
            docs = result.scalars().all()
            return [
                {
                    "id": str(d.id),
                    "filename": d.filename,
                    "file_type": d.file_type,
                    "chunk_count": d.chunk_count,
                    "strategy": d.strategy,
                    "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
                    "status": d.status,
                }
                for d in docs
            ]

    async def delete_document(self, document_id: str) -> bool:
        async with self.session_factory() as session:
            result = await session.execute(
                select(Document).where(Document.id == uuid.UUID(document_id))
            )
            doc = result.scalar_one_or_none()
            if not doc:
                return False
            await session.delete(doc)
            await session.commit()
            return True

    async def count_documents(self) -> int:
        async with self.session_factory() as session:
            result = await session.execute(select(func.count()).select_from(Document))
            return result.scalar() or 0

    async def save_chunk(
        self,
        chunk_id: str,
        document_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        async with self.session_factory() as session:
            chunk = ChunkStore(
                id=chunk_id,
                document_id=document_id,
                text=text,
                metadata_json=metadata,
            )
            session.add(chunk)
            await session.commit()

    async def delete_document_chunks(self, document_id: str) -> None:
        async with self.session_factory() as session:
            result = await session.execute(
                select(ChunkStore).where(ChunkStore.document_id == document_id)
            )
            chunks = result.scalars().all()
            for chunk in chunks:
                await session.delete(chunk)
            await session.commit()

    async def get_all_chunks(self) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            result = await session.execute(select(ChunkStore))
            chunks = result.scalars().all()
            return [
                {
                    "id": c.id,
                    "text": c.text,
                    "metadata": c.metadata_json or {},
                    "document_id": c.document_id,
                }
                for c in chunks
            ]

    async def save_chat_history(
        self,
        session_id: str,
        query: str,
        answer: str,
        citations: list[dict[str, Any]],
        retrieval_scores: list[dict[str, Any]],
        chunks_used: int,
        latency_ms: int,
    ) -> str:
        history_id = str(uuid.uuid4())
        async with self.session_factory() as session:
            record = ChatHistory(
                id=uuid.UUID(history_id),
                session_id=session_id,
                query=query,
                answer=answer,
                citations=citations,
                retrieval_scores=retrieval_scores,
                chunks_used=chunks_used,
                latency_ms=latency_ms,
            )
            session.add(record)
            await session.commit()
        return history_id

    async def get_recent_history(
        self, session_id: str, limit: int = 3
    ) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(ChatHistory)
                .where(ChatHistory.session_id == session_id)
                .order_by(ChatHistory.created_at.desc())
                .limit(limit)
            )
            rows = list(reversed(result.scalars().all()))
            return [
                {
                    "id": str(r.id),
                    "query": r.query,
                    "answer": r.answer,
                    "citations": r.citations,
                    "retrieval_scores": r.retrieval_scores,
                    "chunks_used": r.chunks_used,
                    "latency_ms": r.latency_ms,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    async def get_session_history(
        self, session_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(ChatHistory)
                .where(ChatHistory.session_id == session_id)
                .order_by(ChatHistory.created_at.desc())
                .limit(limit)
            )
            rows = list(reversed(result.scalars().all()))
            return [
                {
                    "id": str(r.id),
                    "query": r.query,
                    "answer": r.answer,
                    "citations": r.citations or [],
                    "retrieval_scores": r.retrieval_scores or [],
                    "chunks_used": r.chunks_used,
                    "latency_ms": r.latency_ms,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    async def get_recent_sessions(self, limit: int = 5) -> list[str]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(ChatHistory.session_id, func.max(ChatHistory.created_at))
                .group_by(ChatHistory.session_id)
                .order_by(func.max(ChatHistory.created_at).desc())
                .limit(limit)
            )
            return [row[0] for row in result.all()]

    async def save_feedback(
        self,
        chat_history_id: str,
        rating: int,
        comment: str | None = None,
    ) -> str:
        feedback_id = str(uuid.uuid4())
        async with self.session_factory() as session:
            fb = Feedback(
                id=uuid.UUID(feedback_id),
                chat_history_id=uuid.UUID(chat_history_id),
                rating=rating,
                comment=comment,
            )
            session.add(fb)
            await session.commit()
        return feedback_id


_postgres: PostgresClient | None = None


async def get_postgres() -> PostgresClient:
    global _postgres
    if _postgres is None:
        _postgres = PostgresClient()
        await _postgres.initialize()
    return _postgres
