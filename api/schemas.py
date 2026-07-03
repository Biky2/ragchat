"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_name: str = Field(default="pasted_text", min_length=1)
    strategy: str = Field(default="fixed")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"fixed", "sentence", "semantic"}
        if v.lower() not in allowed:
            raise ValueError(f"strategy must be one of {allowed}")
        return v.lower()


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    strategy: str
    processing_time_ms: int


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class CitationResponse(BaseModel):
    source_index: int | None = None
    source_file: str
    chunk_index: int | None = None
    excerpt: str
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    chunks_used: int
    retrieval_scores: list[dict[str, Any]]
    latency_ms: int
    session_id: str
    chat_history_id: str | None = None


class FeedbackRequest(BaseModel):
    chat_history_id: str
    rating: int
    comment: str | None = None

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: int) -> int:
        if v not in (1, -1):
            raise ValueError("rating must be 1 or -1")
        return v


class FeedbackResponse(BaseModel):
    success: bool = True


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str | None = None
    chunk_count: int | None = None
    strategy: str | None = None
    uploaded_at: str | None = None
    status: str | None = None


class HistoryItem(BaseModel):
    id: str
    query: str
    answer: str
    citations: list[dict[str, Any]] = []
    retrieval_scores: list[dict[str, Any]] = []
    chunks_used: int | None = None
    latency_ms: int | None = None
    created_at: str | None = None


class HealthResponse(BaseModel):
    postgres: str
    vector_db: str
    ollama: str
    hf_fallback: bool
    embedding_model: str
    total_documents: int


class DeleteResponse(BaseModel):
    success: bool
    document_id: str
