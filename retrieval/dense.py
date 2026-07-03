"""Dense vector similarity retriever (Pinecone or FAISS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import get_settings
from core.vector_store import VectorStore, get_vector_store
from ingestion.embedder import embed_query


@dataclass
class DenseResult:
    id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float


class DenseRetriever:
    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._store = vector_store
        self.settings = get_settings()

    async def _get_store(self) -> VectorStore:
        if self._store is None:
            self._store = await get_vector_store()
        return self._store

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[DenseResult]:
        top_k = top_k or self.settings.dense_top_k
        store = await self._get_store()
        embedding = await embed_query(query)
        results = await store.query(embedding, top_k=top_k)

        return [
            DenseResult(
                id=r.id,
                text=r.text,
                metadata=r.metadata,
                dense_score=r.score,
            )
            for r in results
        ]
