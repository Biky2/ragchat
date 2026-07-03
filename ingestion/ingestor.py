"""Orchestrates load → chunk → embed → upsert pipeline."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from config.settings import get_settings
from core.vector_store import VectorRecord, VectorStore, get_vector_store
from ingestion.chunker import chunk_text
from ingestion.embedder import embed_texts
from ingestion.loader import LoadedDocument, load_from_bytes, load_from_text
from memory.postgres_client import PostgresClient, get_postgres

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_id: str
    filename: str
    chunk_count: int
    strategy: str
    processing_time_ms: int


async def ingest_document(
    content: bytes | None,
    filename: str,
    strategy: str = "fixed",
    raw_text: str | None = None,
    source_name: str | None = None,
    postgres: PostgresClient | None = None,
    vector_store: VectorStore | None = None,
) -> IngestResult:
    start = time.perf_counter()
    settings = get_settings()
    db = postgres or await get_postgres()
    store = vector_store or await get_vector_store()

    document_id = str(uuid.uuid4())

    if raw_text is not None:
        loaded = load_from_text(raw_text, source_name or "pasted_text")
    elif content is not None:
        loaded = load_from_bytes(content, filename)
    else:
        raise ValueError("Either content or raw_text must be provided")

    await db.create_document(
        document_id=document_id,
        filename=loaded.filename,
        file_type=loaded.file_type,
        strategy=strategy,
        status="processing",
    )

    try:
        chunks = chunk_text(loaded.text, loaded.filename, strategy=strategy)
        if not chunks:
            raise ValueError("No chunks produced from document")

        texts = [c.text for c in chunks]
        embeddings = await embed_texts(texts)

        records: list[VectorRecord] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = VectorStore.make_chunk_id(document_id, i)
            metadata = {
                **chunk.metadata,
                "document_id": document_id,
                "text": chunk.text,
            }
            records.append(
                VectorRecord(
                    id=chunk_id,
                    text=chunk.text,
                    embedding=embedding,
                    metadata=metadata,
                )
            )
            await db.save_chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=chunk.text,
                metadata=chunk.metadata,
            )

        await store.upsert(records)
        await db.update_document_status(document_id, "ready", len(chunks))

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return IngestResult(
            document_id=document_id,
            filename=loaded.filename,
            chunk_count=len(chunks),
            strategy=strategy,
            processing_time_ms=elapsed_ms,
        )
    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", document_id, exc)
        await db.update_document_status(document_id, "failed", 0)
        raise


async def delete_document(
    document_id: str,
    postgres: PostgresClient | None = None,
    vector_store: VectorStore | None = None,
) -> bool:
    db = postgres or await get_postgres()
    store = vector_store or await get_vector_store()

    await store.delete_by_document(document_id)
    await db.delete_document_chunks(document_id)
    return await db.delete_document(document_id)
