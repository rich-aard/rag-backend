from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy.orm import Session

from src.api.dependencies import get_llm, get_memory, get_vector_store
from src.core.configs import settings
from src.core.custom_exception import LLMError, MemoryStoreError, VectorStoreError
from src.db.repository import BookingRepository
from src.db.session import get_db
from src.schemas.chat import ChatRequest, ChatResponse
from src.services.booking import handle_booking
from src.services.memory import ChatMemory
from src.services.rag import answer_question
from src.services.vector_store import VectorStore

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
def chat(
    request_data: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
    memory: Annotated[ChatMemory, Depends(get_memory)],
    llm: Annotated[BaseChatModel, Depends(get_llm)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> ChatResponse:
    """Answer a question using conversational RAG and handles bookings."""
    try:
        history = memory.get_history(request_data.session_id)

        response = handle_booking(
            question=request_data.question,
            history=history,
            llm=llm,
            repository=BookingRepository(db),
            session_id=request_data.session_id,
        )
        # rag output
        if response is None:
            result = answer_question(
                question=request_data.question,
                history=history,
                llm=llm,
                vector_store=vector_store,
                top_k=settings.retrieval_top_k,
            )
            response = ChatResponse(
                kind="rag", answer=result.answer, sources=result.sources
            )

        memory.append(
            session_id=request_data.session_id,
            user_message=request_data.question,
            assistant_message=response.answer,
        )

        return response

    except LLMError as exc:
        raise HTTPException(
            status_code=502,
            detail="Language model request failed",
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail="Vector store unavailable",
        ) from exc
    except MemoryStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail="Chat memory unavailable",
        ) from exc
