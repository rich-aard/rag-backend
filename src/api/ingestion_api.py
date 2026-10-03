from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.api.dependencies import get_vector_store
from src.core.configs import settings
from src.core.custom_exception import ExtractionError, VectorStoreError
from src.db.models import Document
from src.db.repository import DocumentRepository
from src.db.session import get_db
from src.schemas.ingestion import (
    ChunkingStrategy,
    DocumentResponse,
    ErrorResponse,
    FileType,
)
from src.services.ingestion import ingest
from src.services.vector_store import VectorStore

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=201,
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def upload_document(
    file: Annotated[UploadFile, File()],
    chunking_strategy: Annotated[ChunkingStrategy, Form()],
    db: Annotated[Session, Depends(get_db)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> Document:
    """Upload and ingest a PDF or text document."""
    filename = file.filename

    if not filename:
        raise HTTPException(
            status_code=422,
            detail="Filename is required",
        )

    extension = Path(filename).suffix.lower().lstrip(".")

    try:
        file_type = FileType(extension)
    except ValueError as exc:
        raise HTTPException(
            status_code=415,
            detail="Only .pdf and .txt files are supported",
        ) from exc

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    content = file.file.read(max_bytes + 1)

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail="File too large",
        )

    try:
        document = ingest(
            filename=filename,
            content=content,
            file_type=file_type,
            strategy=chunking_strategy,
            repository=DocumentRepository(db),
            vector_store=vector_store,
            settings=settings,
        )
    except ExtractionError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail="Vector store unavailable",
        ) from exc

    return document
