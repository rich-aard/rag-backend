import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.chat_api import router as chat_router
from src.api.health import router as health_router
from src.api.ingestion_api import router as ingestion_router
from src.core.configs import settings
from src.db.session import init_db
from src.services.embeddings import create_embeddings
from src.services.llm import create_llm
from src.services.memory import ChatMemory
from src.services.vector_store import VectorStore

logging.basicConfig(
    level=getattr(logging, settings.log_level),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Initialize shared application resources."""
    init_db()

    app.state.embeddings = create_embeddings(
        model_name=settings.embedding_model,
        expected_dimension=settings.embedding_dimension,
    )

    app.state.vector_store = VectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        embeddings=app.state.embeddings,
        dimension=settings.embedding_dimension,
    )
    app.state.llm = create_llm(
        model_name=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )

    app.state.memory = ChatMemory(
        redis_url=settings.redis_url,
        ttl_seconds=settings.chat_history_ttl_seconds,
        max_messages=settings.chat_history_max_messages,
    )
    yield


app = FastAPI(
    title="RAG Backend",
    lifespan=lifespan,
)

app.include_router(
    ingestion_router,
    prefix="/api/v1",
)

app.include_router(health_router, prefix="")
app.include_router(
    chat_router,
    prefix="/api/v1",
)
