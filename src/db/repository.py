from uuid import UUID

from sqlalchemy.orm import Session

from src.db.models import Booking, Document
from src.schemas.ingestion import ChunkingStrategy, FileType


class DocumentRepository:
    """Data access layer for ingested document metadata."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        doc_id: UUID,
        filename: str,
        file_type: FileType,
        chunking_strategy: ChunkingStrategy,
        chunk_count: int,
    ) -> Document:
        """Create a new document record."""
        doc = Document(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type.value,
            chunking_strategy=chunking_strategy.value,
            chunk_count=chunk_count,
        )

        self._session.add(doc)
        self._session.commit()
        self._session.refresh(doc)
        return doc


class BookingRepository:
    """Data access layer for interview bookings."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        session_id: str,
        name: str,
        email: str,
        interview_date: str,
        interview_time: str,
    ) -> Booking:
        """Create and save an interview booking."""
        booking = Booking(
            session_id=session_id,
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
        )

        self._session.add(booking)
        self._session.commit()
        self._session.refresh(booking)

        return booking
