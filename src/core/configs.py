from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # apis
    groq_api_key: SecretStr | None = None

    # models
    llm_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # application
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # vectorstore and database
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    database_url: str = f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}"
    redis_url: str = "redis://localhost:6379/0"

    # chat history values
    chat_history_ttl_seconds: int = Field(default=3600, gt=0)
    chat_history_max_messages: int = Field(default=20, gt=0)

    # ingestion values
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    token_chunk_size: int = Field(default=256, gt=0)
    token_chunk_overlap: int = Field(default=50, ge=0)
    embedding_dimension: int = 384
    max_file_size_mb: int = 50

    # retrieval
    retrieval_top_k: int = Field(default=5, gt=0)

    # timezone
    timezone: str = "UTC"


settings = Settings()
