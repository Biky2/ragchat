"""FastAPI application entry point."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes import _rebuild_bm25_index, router
from config.settings import get_settings
from core.vector_store import get_vector_store
from memory.postgres_client import get_postgres

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting RAGChat application...")

    try:
        await get_postgres()
    except Exception as exc:  
        logger.warning(
            "Postgres initialization failed; continuing without DB: %s", exc
        )
    await get_vector_store()
    await _rebuild_bm25_index()
    logger.info("Application startup complete")
    yield


    try:
        from memory.postgres_client import _postgres as _pg

        if _pg is not None:
            await _pg.close()
    except Exception:
        logger.debug("Postgres close skipped or failed during shutdown")
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAGChat",
        description="Enterprise Intelligent RAG-Based Chatbot",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    static_dir = settings.static_dir
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)
