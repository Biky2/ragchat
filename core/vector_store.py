"""Unified vector store: Pinecone primary, FAISS local fallback."""

from __future__ import annotations

import json
import logging
import pickle
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


@dataclass
class VectorRecord:
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class VectorStore:
    """Abstracts Pinecone and FAISS behind a single async API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._backend: str = "faiss"
        self._pinecone_index: Any = None
        self._faiss_index: Any = None
        self._id_map: dict[int, str] = {}
        self._records: dict[str, VectorRecord] = {}
        self._initialized = False

    @property
    def backend(self) -> str:
        return self._backend

    async def initialize(self) -> None:
        if self._initialized:
            return

        if self.settings.pinecone_configured:
            try:
                await self._init_pinecone()
                self._backend = "pinecone"
                logger.info("Vector store initialized with Pinecone")
                self._initialized = True
                return
            except Exception as exc:
                logger.warning("Pinecone unavailable, falling back to FAISS: %s", exc)

        await self._init_faiss()
        self._backend = "faiss"
        logger.info("Vector store initialized with FAISS at %s", self.settings.faiss_path)
        self._initialized = True

    async def _init_pinecone(self) -> None:
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=self.settings.pinecone_api_key)
        index_name = self.settings.pinecone_index_name

        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.settings.pinecone_environment.replace("-aws", ""),
                ),
            )

        self._pinecone_index = pc.Index(index_name)

    async def _init_faiss(self) -> None:
        import faiss

        self.settings.faiss_path.mkdir(parents=True, exist_ok=True)
        index_file = self.settings.faiss_path / "index.faiss"
        meta_file = self.settings.faiss_path / "metadata.pkl"

        if index_file.exists() and meta_file.exists():
            self._faiss_index = faiss.read_index(str(index_file))
            with open(meta_file, "rb") as f:
                data = pickle.load(f)
                self._id_map = data.get("id_map", {})
                self._records = {
                    k: VectorRecord(**v) if isinstance(v, dict) else v
                    for k, v in data.get("records", {}).items()
                }
        else:
            self._faiss_index = faiss.IndexFlatIP(384)
            self._id_map = {}
            self._records = {}

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    async def upsert(self, records: list[VectorRecord]) -> int:
        await self.initialize()
        if not records:
            return 0

        total = 0
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            if self._backend == "pinecone":
                await self._upsert_pinecone(batch)
            else:
                await self._upsert_faiss(batch)
            total += len(batch)
        return total

    async def _upsert_pinecone(self, batch: list[VectorRecord]) -> None:
        vectors = [
            {
                "id": rec.id,
                "values": rec.embedding,
                "metadata": {
                    **rec.metadata,
                    "text": rec.text[:1000],
                },
            }
            for rec in batch
        ]
        self._pinecone_index.upsert(vectors=vectors)

    async def _upsert_faiss(self, batch: list[VectorRecord]) -> None:
        import faiss

        vectors = np.array([rec.embedding for rec in batch], dtype=np.float32)
        vectors = self._normalize(vectors)

        start_idx = self._faiss_index.ntotal
        self._faiss_index.add(vectors)

        for offset, rec in enumerate(batch):
            idx = start_idx + offset
            self._id_map[idx] = rec.id
            self._records[rec.id] = rec

        await self._save_faiss()

    async def _save_faiss(self) -> None:
        import faiss

        index_file = self.settings.faiss_path / "index.faiss"
        meta_file = self.settings.faiss_path / "metadata.pkl"

        faiss.write_index(self._faiss_index, str(index_file))
        serializable_records = {
            k: {
                "id": v.id,
                "text": v.text,
                "embedding": v.embedding,
                "metadata": v.metadata,
            }
            for k, v in self._records.items()
        }
        with open(meta_file, "wb") as f:
            pickle.dump({"id_map": self._id_map, "records": serializable_records}, f)

    async def query(
        self,
        embedding: list[float],
        top_k: int = 20,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        await self.initialize()

        if self._backend == "pinecone":
            return await self._query_pinecone(embedding, top_k, filter_metadata)
        return await self._query_faiss(embedding, top_k, filter_metadata)

    async def _query_pinecone(
        self,
        embedding: list[float],
        top_k: int,
        filter_metadata: dict[str, Any] | None,
    ) -> list[VectorSearchResult]:
        query_kwargs: dict[str, Any] = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True,
        }
        if filter_metadata:
            query_kwargs["filter"] = filter_metadata

        response = self._pinecone_index.query(**query_kwargs)
        results: list[VectorSearchResult] = []
        for match in response.get("matches", []):
            meta = dict(match.get("metadata") or {})
            text = meta.pop("text", "")
            results.append(
                VectorSearchResult(
                    id=match["id"],
                    text=text,
                    metadata=meta,
                    score=float(match.get("score", 0.0)),
                )
            )
        return results

    async def _query_faiss(
        self,
        embedding: list[float],
        top_k: int,
        filter_metadata: dict[str, Any] | None,
    ) -> list[VectorSearchResult]:
        if self._faiss_index.ntotal == 0:
            return []

        vector = np.array([embedding], dtype=np.float32)
        vector = self._normalize(vector)
        k = min(top_k, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(vector, k)

        results: list[VectorSearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record_id = self._id_map.get(int(idx))
            if not record_id or record_id not in self._records:
                continue
            rec = self._records[record_id]
            if filter_metadata:
                if not all(rec.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            results.append(
                VectorSearchResult(
                    id=rec.id,
                    text=rec.text,
                    metadata=rec.metadata,
                    score=float(score),
                )
            )
        return results[:top_k]

    async def delete_by_document(self, document_id: str) -> int:
        await self.initialize()
        if self._backend == "pinecone":
            return await self._delete_pinecone_document(document_id)
        return await self._delete_faiss_document(document_id)

    async def _delete_pinecone_document(self, document_id: str) -> int:
        try:
            self._pinecone_index.delete(filter={"document_id": document_id})
            return 1
        except Exception as exc:
            logger.error("Failed to delete from Pinecone: %s", exc)
            return 0

    async def _delete_faiss_document(self, document_id: str) -> int:
        import faiss

        to_remove = [
            rid for rid, rec in self._records.items()
            if rec.metadata.get("document_id") == document_id
        ]
        if not to_remove:
            return 0

        remaining = [
            rec for rid, rec in self._records.items() if rid not in to_remove
        ]
        self._records = {rec.id: rec for rec in remaining}
        self._faiss_index = faiss.IndexFlatIP(384)
        self._id_map = {}

        if remaining:
            vectors = np.array([rec.embedding for rec in remaining], dtype=np.float32)
            vectors = self._normalize(vectors)
            self._faiss_index.add(vectors)
            for idx, rec in enumerate(remaining):
                self._id_map[idx] = rec.id

        await self._save_faiss()
        return len(to_remove)

    async def get_total_vectors(self) -> int:
        await self.initialize()
        if self._backend == "pinecone":
            stats = self._pinecone_index.describe_index_stats()
            return int(stats.get("total_vector_count", 0))
        return self._faiss_index.ntotal if self._faiss_index else 0

    @staticmethod
    def make_chunk_id(document_id: str, chunk_index: int) -> str:
        return f"{document_id}_{chunk_index}"

    @staticmethod
    def new_document_id() -> str:
        return str(uuid.uuid4())


_vector_store: VectorStore | None = None


async def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        await _vector_store.initialize()
    return _vector_store
