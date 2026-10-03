import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


class Document(Base):
    """Database model for ingested document metadata."""

    __tablename__ = "documents"

    doc_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    filename: Mapped[str] = mapped_column()
    file_type: Mapped[str] = mapped_column()
    chunking_strategy: Mapped[str] = mapped_column()
    chunk_count: Mapped[int] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC),
    )


class Booking(Base):
    """Database model for interview bookings."""

    __tablename__ = "bookings"

    booking_id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[str] = mapped_column()
    name: Mapped[str] = mapped_column()
    email: Mapped[str] = mapped_column()
    interview_date: Mapped[str] = mapped_column()
    interview_time: Mapped[str] = mapped_column()
    status: Mapped[str] = mapped_column(default="confirmed")
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC),
    )
