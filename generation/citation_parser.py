"""Extract and map [Source N] citations from LLM output."""

from __future__ import annotations

import re
from typing import Any

from retrieval.hybrid import HybridResult

CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


def parse_citations(
    answer: str,
    chunks: list[HybridResult],
) -> list[dict[str, Any]]:
    matches = CITATION_PATTERN.findall(answer)
    seen: set[int] = set()
    citations: list[dict[str, Any]] = []

    for match in matches:
        idx = int(match) - 1
        if idx in seen or idx < 0 or idx >= len(chunks):
            continue
        seen.add(idx)
        chunk = chunks[idx]
        excerpt = chunk.text[:300] + ("..." if len(chunk.text) > 300 else "")
        citations.append(
            {
                "source_index": idx + 1,
                "source_file": chunk.metadata.get("source_file", "unknown"),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "excerpt": excerpt,
                "dense_score": chunk.dense_score,
                "sparse_score": chunk.sparse_score,
                "rrf_score": chunk.rrf_score,
            }
        )

    return citations


def extract_source_numbers(answer: str) -> list[int]:
    return [int(m) for m in CITATION_PATTERN.findall(answer)]
