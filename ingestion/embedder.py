"""Async-safe sentence-transformers embedding wrapper."""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Sequence

import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)

_model = None
_model_lock = asyncio.Lock()


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed_texts_sync(texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
    if not texts:
        return []

    model = _load_model()
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()
    return [e.tolist() for e in embeddings]


async def embed_texts(texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: embed_texts_sync(texts, batch_size)
    )


async def embed_query(query: str) -> list[float]:
    results = await embed_texts([query])
    return results[0]


async def check_embedding_model() -> bool:
    try:
        await embed_texts(["health check"])
        return True
    except Exception as exc:
        logger.error("Embedding model check failed: %s", exc)
        return False
