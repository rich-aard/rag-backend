from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import SecretStr


def create_llm(
    model_name: str,
    api_key: SecretStr | None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Create the configured Groq chat model."""
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")

    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )
