"""FastAPI route handlers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    HistoryItem,
    IngestResponse,
    IngestTextRequest,
)
from config.settings import get_settings
from core.llm import check_huggingface_health, check_ollama_health
from core.vector_store import get_vector_store
from generation.chain import RAGChain
from ingestion.embedder import check_embedding_model
from ingestion.ingestor import delete_document, ingest_document
from memory.postgres_client import get_postgres
from retrieval.sparse import get_sparse_retriever

logger = logging.getLogger(__name__)
router = APIRouter()
rag_chain = RAGChain()


async def _rebuild_bm25_index() -> None:
    db = await get_postgres()
    chunks = await db.get_all_chunks()
    sparse = get_sparse_retriever()
    sparse.build_index(chunks)
    logger.info("BM25 index rebuilt with %d chunks", sparse.chunk_count)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    strategy: str = Form(default="fixed"),
):
    settings = get_settings()

    if strategy.lower() not in ("fixed", "sentence", "semantic"):
        raise HTTPException(status_code=400, detail="Invalid strategy")

    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {settings.allowed_extensions}",
        )

    content = await file.read()
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds {settings.max_upload_size_mb}MB limit",
        )

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        result = await ingest_document(content=content, filename=filename, strategy=strategy.lower())
        await _rebuild_bm25_index()
        return IngestResponse(
            document_id=result.document_id,
            filename=result.filename,
            chunk_count=result.chunk_count,
            strategy=result.strategy,
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(body: IngestTextRequest):
    try:
        result = await ingest_document(
            content=None,
            filename=body.source_name,
            strategy=body.strategy,
            raw_text=body.text,
            source_name=body.source_name,
        )
        await _rebuild_bm25_index()
        return IngestResponse(
            document_id=result.document_id,
            filename=result.filename,
            chunk_count=result.chunk_count,
            strategy=result.strategy,
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as exc:
        logger.exception("Text ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    try:
        result = await rag_chain.run(
            query=body.query,
            session_id=body.session_id,
            top_k=body.top_k,
        )
        return ChatResponse(
            answer=result.answer,
            citations=result.citations,
            chunks_used=result.chunks_used,
            retrieval_scores=result.retrieval_scores,
            latency_ms=result.latency_ms,
            session_id=result.session_id,
            chat_history_id=result.chat_history_id,
        )
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/chat/stream/{session_id}")
async def chat_stream(session_id: str, query: str, top_k: int = 5):
    if not query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    async def event_generator():
        try:
            async for event in rag_chain.stream(query=query, session_id=session_id, top_k=top_k):
                yield {"event": "message", "data": json.dumps(event)}
        except Exception as exc:
            logger.exception("Stream failed")
            yield {
                "event": "message",
                "data": json.dumps({"token": "", "done": True, "error": str(exc)}),
            }

    return EventSourceResponse(event_generator())


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(body: FeedbackRequest):
    db = await get_postgres()
    try:
        await db.save_feedback(
            chat_history_id=body.chat_history_id,
            rating=body.rating,
            comment=body.comment,
        )
        return FeedbackResponse(success=True)
    except Exception as exc:
        logger.exception("Feedback failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents():
    db = await get_postgres()
    docs = await db.list_documents()
    return [DocumentResponse(**d) for d in docs]


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def remove_document(document_id: str):
    success = await delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    await _rebuild_bm25_index()
    return DeleteResponse(success=True, document_id=document_id)


@router.get("/history/{session_id}", response_model=list[HistoryItem])
async def get_history(session_id: str):
    db = await get_postgres()
    history = await db.get_session_history(session_id, limit=20)
    return [HistoryItem(**h) for h in history]


@router.get("/sessions")
async def get_recent_sessions():
    db = await get_postgres()
    sessions = await db.get_recent_sessions(limit=5)
    return {"sessions": sessions}


@router.get("/health", response_model=HealthResponse)
async def health():
    db = await get_postgres()
    store = await get_vector_store()

    postgres_ok = await db.health_check()
    embedding_ok = await check_embedding_model()
    ollama_ok = await check_ollama_health()
    hf_ok = await check_huggingface_health()
    total_docs = await db.count_documents()

    return HealthResponse(
        postgres="ok" if postgres_ok else "unavailable",
        vector_db=store.backend,
        ollama="ok" if ollama_ok else "unavailable",
        hf_fallback=hf_ok,
        embedding_model="ok" if embedding_ok else "unavailable",
        total_documents=total_docs,
    )
