import logging
import uuid

from src.core.configs import Settings
from src.core.custom_exception import ExtractionError
from src.db.models import Document
from src.db.repository import DocumentRepository
from src.schemas.ingestion import ChunkingStrategy, FileType
from src.services.chunking import get_splitter
from src.services.extraction import extract_text
from src.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest(
    filename: str,
    content: bytes,
    file_type: FileType,
    strategy: ChunkingStrategy,
    repository: DocumentRepository,
    vector_store: VectorStore,
    settings: Settings,
) -> Document:
    """Extract, chunk, embed, and store a document."""
    text = extract_text(content, file_type)

    splitter = get_splitter(
        strategy=strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        token_chunk_size=settings.token_chunk_size,
        token_chunk_overlap=settings.token_chunk_overlap,
    )

    chunks = [c for c in splitter.split_text(text) if c and c.strip()]

    if not chunks:
        raise ExtractionError("No text chunks could be created")

    doc_id = uuid.uuid4()

    try:
        vector_store.add_documents(
            chunks=chunks,
            doc_id=str(doc_id),
            filename=filename,
        )

        document = repository.create(
            filename=filename,
            file_type=file_type,
            chunking_strategy=strategy,
            chunk_count=len(chunks),
            doc_id=doc_id,
        )

    except Exception:
        logger.exception("Ingestion failed for %s", filename)
        raise

    logger.info("Ingested %s as %s chunks", filename, len(chunks))

    return document
