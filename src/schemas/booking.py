from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class BookingStatus(StrEnum):
    CONFIRMED = "confirmed"


class BookingInfo(BaseModel):
    """Booking informations extracted from the conversation."""

    is_booking_request: bool = False
    name: str | None = None
    email: str | None = None
    interview_date: str | None = Field(
        default=None, description="Interview date as YYYY-MM-DD"
    )
    interview_time: str | None = Field(
        default=None, description="Interview time in 24-hour HH:MM"
    )


class BookingResponse(BaseModel):
    """Response returned after booking an interview."""

    model_config = ConfigDict(from_attributes=True)

    booking_id: UUID
    session_id: str
    name: str
    email: str
    interview_date: str
    interview_time: str
    status: BookingStatus
    created_at: datetime
