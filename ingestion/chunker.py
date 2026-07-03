"""Custom chunking strategies: fixed, sentence, and semantic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import nltk

from config.settings import get_settings

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]


def _estimate_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _make_metadata(
    source_file: str,
    chunk_index: int,
    strategy: str,
    char_start: int,
    char_end: int,
    text: str,
) -> dict[str, Any]:
    return {
        "source_file": source_file,
        "chunk_index": chunk_index,
        "strategy": strategy,
        "char_start": char_start,
        "char_end": char_end,
        "token_count": _estimate_tokens(text),
    }


def chunk_fixed(
    text: str,
    source_file: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start_word = 0
    chunk_index = 0

    while start_word < len(words):
        end_word = min(start_word + chunk_size, len(words))
        chunk_words = words[start_word:end_word]
        chunk_text = " ".join(chunk_words)

        char_start = len(" ".join(words[:start_word]))
        if start_word > 0:
            char_start += 1
        char_end = char_start + len(chunk_text)

        chunks.append(
            Chunk(
                text=chunk_text,
                metadata=_make_metadata(
                    source_file, chunk_index, ChunkStrategy.FIXED.value,
                    char_start, char_end, chunk_text,
                ),
            )
        )
        chunk_index += 1

        if end_word >= len(words):
            break
        start_word = end_word - chunk_overlap
        if start_word <= 0:
            start_word = end_word

    return chunks


def chunk_sentence(
    text: str,
    source_file: str,
    target_tokens: int = 400,
    overlap_sentences: int = 2,
) -> list[Chunk]:
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_index = 0
    char_offset = 0

    def flush(sents: list[str], idx: int) -> None:
        nonlocal char_offset
        if not sents:
            return
        chunk_text = " ".join(sents)
        char_start = text.find(sents[0], char_offset)
        if char_start == -1:
            char_start = char_offset
        char_end = char_start + len(chunk_text)
        chunks.append(
            Chunk(
                text=chunk_text,
                metadata=_make_metadata(
                    source_file, idx, ChunkStrategy.SENTENCE.value,
                    char_start, char_end, chunk_text,
                ),
            )
        )
        char_offset = char_end

    for sentence in sentences:
        sent_tokens = _estimate_tokens(sentence)
        if current_tokens + sent_tokens > target_tokens and current_sentences:
            flush(current_sentences, chunk_index)
            chunk_index += 1
            overlap = current_sentences[-overlap_sentences:] if overlap_sentences else []
            current_sentences = overlap + [sentence]
            current_tokens = sum(_estimate_tokens(s) for s in current_sentences)
        else:
            current_sentences.append(sentence)
            current_tokens += sent_tokens

    if current_sentences:
        flush(current_sentences, chunk_index)

    return chunks


def chunk_semantic(
    text: str,
    source_file: str,
    threshold: float = 0.75,
    max_tokens: int = 600,
    embed_fn=None,
) -> list[Chunk]:
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        s = sentences[0]
        return [
            Chunk(
                text=s,
                metadata=_make_metadata(
                    source_file, 0, ChunkStrategy.SEMANTIC.value,
                    0, len(s), s,
                ),
            )
        ]

    if embed_fn is None:
        from ingestion.embedder import embed_texts_sync
        embeddings = embed_texts_sync(sentences)
    else:
        embeddings = embed_fn(sentences)

    import numpy as np

    def cosine_sim(a, b) -> float:
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        current_tokens = sum(_estimate_tokens(s) for s in groups[-1])
        new_tokens = current_tokens + _estimate_tokens(sentences[i])

        if sim >= threshold and new_tokens <= max_tokens:
            groups[-1].append(sentences[i])
        else:
            groups.append([sentences[i]])

    chunks: list[Chunk] = []
    search_start = 0
    for chunk_index, group in enumerate(groups):
        chunk_text = " ".join(group)
        char_start = text.find(group[0], search_start)
        if char_start == -1:
            char_start = search_start
        char_end = char_start + len(chunk_text)
        search_start = char_end

        chunks.append(
            Chunk(
                text=chunk_text,
                metadata=_make_metadata(
                    source_file, chunk_index, ChunkStrategy.SEMANTIC.value,
                    char_start, char_end, chunk_text,
                ),
            )
        )

    return chunks


def chunk_text(
    text: str,
    source_file: str,
    strategy: str = "fixed",
    **kwargs,
) -> list[Chunk]:
    strategy_enum = ChunkStrategy(strategy.lower())

    if strategy_enum == ChunkStrategy.FIXED:
        return chunk_fixed(text, source_file, **kwargs)
    if strategy_enum == ChunkStrategy.SENTENCE:
        return chunk_sentence(text, source_file, **kwargs)
    if strategy_enum == ChunkStrategy.SEMANTIC:
        return chunk_semantic(text, source_file, **kwargs)

    raise ValueError(f"Unknown chunking strategy: {strategy}")
