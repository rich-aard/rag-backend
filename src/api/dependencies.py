from fastapi import Request
from langchain_core.language_models.chat_models import BaseChatModel

from src.services.memory import ChatMemory
from src.services.vector_store import VectorStore


def get_memory(request: Request) -> ChatMemory:
    """Get the shared chat memory from application state."""
    memory: ChatMemory = request.app.state.memory
    return memory


def get_llm(request: Request) -> BaseChatModel:
    """Get the shared language model from application state."""
    llm: BaseChatModel = request.app.state.llm
    return llm


def get_vector_store(request: Request) -> VectorStore:
    """Get the shared vector store from application state."""
    vector_store: VectorStore = request.app.state.vector_store
    return vector_store
