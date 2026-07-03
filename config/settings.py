"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Pinecone
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="ragchat", alias="PINECONE_INDEX_NAME")
    pinecone_environment: str = Field(default="us-east-1-aws", alias="PINECONE_ENVIRONMENT")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/ragchat",
        alias="DATABASE_URL",
    )

    # LLM
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    llm_model: str = Field(default="llama3.2", alias="LLM_MODEL")
    huggingface_api_key: str = Field(default="", alias="HUGGINGFACE_API_KEY")
    huggingface_model: str = Field(
        default="Qwen/Qwen3-32B",
        alias="HUGGINGFACE_MODEL",
    )

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    # FAISS
    faiss_index_path: str = Field(default="./data/faiss_index", alias="FAISS_INDEX_PATH")

    # Chunking defaults
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")

    # Upload limits
    max_upload_size_mb: int = Field(default=50, alias="MAX_UPLOAD_SIZE_MB")
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt")

    # Retrieval
    dense_top_k: int = Field(default=20, alias="DENSE_TOP_K")
    sparse_top_k: int = Field(default=20, alias="SPARSE_TOP_K")
    rrf_k: int = Field(default=60, alias="RRF_K")
    default_top_k: int = Field(default=5, alias="DEFAULT_TOP_K")

    # Session memory
    max_session_exchanges: int = Field(default=6, alias="MAX_SESSION_EXCHANGES")
    context_exchanges: int = Field(default=3, alias="CONTEXT_EXCHANGES")

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    static_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "static")
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def faiss_path(self) -> Path:
        path = Path(self.faiss_index_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    @property
    def pinecone_configured(self) -> bool:
        return bool(self.pinecone_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
