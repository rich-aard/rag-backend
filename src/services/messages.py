from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.schemas.chat import ChatMessage


def to_langchain_messages(history: list[ChatMessage]) -> list[BaseMessage]:
    """Convert application chat messages to LangChain messages."""
    return [
        HumanMessage(content=m.content)
        if m.role == "user"
        else AIMessage(content=m.content)
        for m in history
    ]
