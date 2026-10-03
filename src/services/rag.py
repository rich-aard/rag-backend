import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.core.custom_exception import LLMError
from src.schemas.chat import ChatMessage, RagResult
from src.services.messages import to_langchain_messages
from src.services.prompts import answer_prompt, rewrite_prompt
from src.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _rewrite_question(
    question: str,
    history: list[ChatMessage],
    llm: BaseChatModel,
) -> str:
    """Rewrite a follow-up question into a standalone question."""
    if not history:
        return question

    chain = rewrite_prompt | llm | StrOutputParser()
    try:
        rewritten_question = chain.invoke(
            {"chat_history": to_langchain_messages(history[-6:]), "input": question}
        )
    except Exception as exc:
        logger.exception("Question rewrite failed")
        raise LLMError("The language model request failed") from exc

    return rewritten_question.strip() or question


def answer_question(
    question: str,
    history: list[ChatMessage],
    llm: BaseChatModel,
    vector_store: VectorStore,
    top_k: int,
) -> RagResult:
    """Answer a question using conversational RAG."""
    standalone_question = _rewrite_question(
        question=question,
        history=history,
        llm=llm,
    )

    sources = vector_store.search(
        query=standalone_question,
        top_k=top_k,
    )

    if not sources:
        return RagResult(
            answer="I don't know based on the provided document.",
            sources=[],
        )

    context = "\n\n".join(
        f"Source: {source.filename}\n{source.text}" for source in sources
    )

    chain = answer_prompt | llm | StrOutputParser()
    try:
        answer = chain.invoke(
            {
                "context": context,
                "chat_history": to_langchain_messages(history),
                "input": question,
            }
        )
    except Exception as exc:
        logger.exception("Answer generation failed")

        raise LLMError("The language model request failed") from exc

    return RagResult(
        answer=answer.strip(),
        sources=sources,
    )
