"""BM25 sparse retriever using rank-bm25."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from config.settings import get_settings


@dataclass
class SparseResult:
    id: str
    text: str
    metadata: dict[str, Any]
    sparse_score: float


class SparseRetriever:
    """In-memory BM25 index rebuilt from stored chunks."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._chunks: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self._tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[SparseResult]:
        top_k = top_k or self.settings.sparse_top_k

        if not self._bm25 or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: list[SparseResult] = []
        for idx in scored_indices:
            if scores[idx] <= 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                SparseResult(
                    id=chunk["id"],
                    text=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                    sparse_score=float(scores[idx]),
                )
            )

        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


_sparse_retriever: SparseRetriever | None = None


def get_sparse_retriever() -> SparseRetriever:
    global _sparse_retriever
    if _sparse_retriever is None:
        _sparse_retriever = SparseRetriever()
    return _sparse_retriever
