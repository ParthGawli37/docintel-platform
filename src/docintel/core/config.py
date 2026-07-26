"""
Central application configuration.

All runtime configuration is sourced from environment variables / .env.
No values are hardcoded or guessed here — fields with no sensible default
are left unset and validated at startup, so misconfiguration fails fast
and loudly rather than silently falling back to an invented value.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LogFormat(StrEnum):
    JSON = "json"
    CONSOLE = "console"


class QdrantMode(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class StorageBackend(StrEnum):
    LOCAL = "local"
    S3 = "s3"


class ChunkStrategy(StrEnum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"


class Settings(BaseSettings):
    """
    Single source of truth for configuration.

    Instantiate once via `get_settings()` (see below); do not construct
    ad hoc elsewhere, so the whole app shares one validated config object.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App ----
    app_name: str = "docintel"
    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    log_format: LogFormat = LogFormat.JSON

    # ---- NVIDIA API ----
    nvidia_api_key: str = Field(..., description="Required. No default — must be supplied.")
    nvidia_api_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_generation_model: str = Field(
        ...,
        description="TODO(user): exact NVIDIA generation model ID. Not guessed by this platform.",
    )
    nvidia_embedding_model: str = Field(
        ...,
        description="TODO(user): exact NVIDIA embedding model ID. Not guessed by this platform.",
    )
    nvidia_embedding_dimensions: int = Field(
        ..., description="Must match the embedding model above."
    )

    # ---- Qdrant ----
    qdrant_mode: QdrantMode = QdrantMode.LOCAL
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # ---- Storage layer ----
    storage_backend: StorageBackend = StorageBackend.LOCAL
    raw_files_dir: Path = Path("./data/raw")
    cache_dir: Path = Path("./data/cache")
    hash_registry_path: Path = Path("./data/cache/hash_registry.sqlite")

    # ---- Ingestion ----
    ocr_engine: str = "tesseract"
    max_upload_file_mb: int = 50

    # ---- Chunking ----
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # ---- Retrieval ----
    hybrid_search_alpha: float = Field(0.5, ge=0.0, le=1.0)
    rerank_top_k: int = 10
    retrieval_top_k: int = 5

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: str = "*"

    def model_post_init(self, __context: object, /) -> None:
        if self.qdrant_mode is QdrantMode.CLOUD and not self.qdrant_api_key:
            raise ValueError(
                "QDRANT_MODE=cloud requires QDRANT_API_KEY to be set. "
                "Refusing to silently fall back to local mode."
            )
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS."
            )
        self.raw_files_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hash_registry_path.parent.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Return the process-wide Settings singleton, constructing it on first call.

    Using a function (rather than a module-level instance) means import
    time never triggers env validation — useful for tests that need to
    monkeypatch environment variables before Settings is built.
    """
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]  # values come from env
    return _settings


def reset_settings_cache() -> None:
    """Test helper: force get_settings() to rebuild on next call."""
    global _settings
    _settings = None
