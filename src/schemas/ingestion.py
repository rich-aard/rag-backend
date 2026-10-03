from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ChunkingStrategy(StrEnum):
    """Chunking strategies for the documents."""

    RECURSIVE = "recursive"
    TOKEN = "token"


class FileType(StrEnum):
    """Supported file types."""

    PDF = "pdf"
    TXT = "txt"


class DocumentResponse(BaseModel):
    """Defines responses obtained after successfull document ingestion."""

    model_config = ConfigDict(from_attributes=True)

    doc_id: UUID
    filename: str
    file_type: FileType
    chunking_strategy: ChunkingStrategy
    chunk_count: int
    created_at: datetime


class ErrorResponse(BaseModel):
    """Returns API error response."""

    detail: str
