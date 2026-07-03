"""Hybrid retrieval via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import get_settings
from retrieval.dense import DenseRetriever, DenseResult
from retrieval.sparse import SparseRetriever, get_sparse_retriever


@dataclass
class HybridResult:
    id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float | None
    sparse_score: float | None
    rrf_score: float


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever | None = None,
        sparse: SparseRetriever | None = None,
    ) -> None:
        self.settings = get_settings()
        self.dense = dense or DenseRetriever()
        self.sparse = sparse or get_sparse_retriever()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[HybridResult]:
        top_k = top_k or self.settings.default_top_k
        dense_top_k = self.settings.dense_top_k
        sparse_top_k = self.settings.sparse_top_k
        k = self.settings.rrf_k

        dense_results: list[DenseResult] = await self.dense.retrieve(query, dense_top_k)
        sparse_results: list[SparseResult] = self.sparse.retrieve(query, sparse_top_k)

        dense_ranks: dict[str, int] = {r.id: i + 1 for i, r in enumerate(dense_results)}
        sparse_ranks: dict[str, int] = {r.id: i + 1 for i, r in enumerate(sparse_results)}

        dense_map: dict[str, DenseResult] = {r.id: r for r in dense_results}
        sparse_map: dict[str, SparseResult] = {r.id: r for r in sparse_results}

        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        fused: list[HybridResult] = []

        for chunk_id in all_ids:
            rrf_score = 0.0
            if chunk_id in dense_ranks:
                rrf_score += 1.0 / (k + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                rrf_score += 1.0 / (k + sparse_ranks[chunk_id])

            dense_r = dense_map.get(chunk_id)
            sparse_r = sparse_map.get(chunk_id)

            text = (dense_r.text if dense_r else sparse_r.text)  # type: ignore[union-attr]
            metadata = (
                dense_r.metadata if dense_r else sparse_r.metadata  # type: ignore[union-attr]
            )

            fused.append(
                HybridResult(
                    id=chunk_id,
                    text=text,
                    metadata=metadata,
                    dense_score=dense_r.dense_score if dense_r else None,
                    sparse_score=sparse_r.sparse_score if sparse_r else None,
                    rrf_score=rrf_score,
                )
            )

        fused.sort(key=lambda x: x.rrf_score, reverse=True)
        return fused[:top_k]
