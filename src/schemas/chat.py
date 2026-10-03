from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.booking import BookingResponse


class ChatMessage(BaseModel):
    """A message stored in conversation history."""

    role: Literal["user", "assistant"]
    content: str


class RetrievedChunk(BaseModel):
    """Chunks based on query."""

    doc_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class RagResult(BaseModel):
    """Answer obtained from RAG."""

    answer: str
    sources: list[RetrievedChunk]


class ChatRequest(BaseModel):
    """Request body for a chat message."""

    session_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    """Response body from chat."""

    kind: Literal["rag", "booking"]
    answer: str
    sources: list[RetrievedChunk] = []
    booking: BookingResponse | None = None
