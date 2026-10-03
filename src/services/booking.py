import logging
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import EmailStr, TypeAdapter, ValidationError

from src.core.configs import settings
from src.core.custom_exception import LLMError
from src.db.repository import BookingRepository
from src.schemas.booking import BookingInfo, BookingResponse
from src.schemas.chat import ChatMessage, ChatResponse
from src.services.messages import to_langchain_messages
from src.services.prompts import booking_prompt

logger = logging.getLogger(__name__)
_email_adapter = TypeAdapter(EmailStr)
_REQUIRED_FIELD_LABELS = {
    "name": "your full name",
    "email": "your email address",
    "interview_date": "the interview date",
    "interview_time": "the interview time",
}


def _reply(text: str) -> ChatResponse:
    return ChatResponse(kind="booking", answer=text)


def extract_booking(
    question: str,
    history: list[ChatMessage],
    llm: BaseChatModel,
) -> BookingInfo:
    """Extract interview booking information using the LLM.
    If the model fails to return structured output, the message is treated
    as a normal question. A real LLM outage still fails the answer step.
    """

    chain = booking_prompt | llm.with_structured_output(BookingInfo)

    try:
        result = chain.invoke(
            {
                "chat_history": to_langchain_messages(history),
                "input": question,
                "today": datetime.now(ZoneInfo(settings.timezone)).date().isoformat(),
            }
        )

    except Exception:
        logger.exception("Booking extraction failed")
        return BookingInfo()

    if not isinstance(result, BookingInfo):
        raise LLMError("Could not extract booking information")
    return result


def handle_booking(
    question: str,
    history: list[ChatMessage],
    llm: BaseChatModel,
    repository: BookingRepository,
    session_id: str,
) -> ChatResponse | None:
    """Handles booking for one turn."""

    info = extract_booking(
        question=question,
        history=history,
        llm=llm,
    )

    if not info.is_booking_request:
        return None

    email: str | None = None
    if info.email:
        try:
            email = _email_adapter.validate_python(info.email)
        except ValidationError:
            return _reply("That email address doesn't look valid. Could you check it?")

    if not (info.name and email and info.interview_date and info.interview_time):
        missing = ", ".join(
            label
            for field, label in _REQUIRED_FIELD_LABELS.items()
            if not getattr(info, field)
        )

        return _reply(
            f"Yes, I can book your interview. Please provide the following details: {missing}."
        )

    try:
        interview_date = date.fromisoformat(info.interview_date)
        interview_time = time.fromisoformat(info.interview_time)
    except ValueError:
        return _reply(
            "I couldn't understand the date or time. "
            "Please give them like 2026-10-12 and 15:00."
        )

    if interview_date < datetime.now(ZoneInfo(settings.timezone)).date():
        return _reply("That date has already passed. Please choose a future date.")

    saved_booking = repository.create(
        session_id=session_id,
        name=info.name,
        email=email,
        interview_date=interview_date.isoformat(),
        interview_time=interview_time.strftime("%H:%M"),
    )
    return ChatResponse(
        kind="booking",
        answer=(
            f"Your interview is booked for {saved_booking.interview_date} at {saved_booking.interview_time}."
        ),
        booking=BookingResponse.model_validate(saved_booking),
    )
