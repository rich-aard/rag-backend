from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

from src.schemas.ingestion import ChunkingStrategy


def get_splitter(
    strategy: ChunkingStrategy,
    chunk_size: int,
    chunk_overlap: int,
    token_chunk_size: int,
    token_chunk_overlap: int,
) -> TextSplitter:
    """Returns the LangChain splitter for the requested strategy."""
    if strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    if strategy == ChunkingStrategy.TOKEN:
        return TokenTextSplitter(
            chunk_size=token_chunk_size, chunk_overlap=token_chunk_overlap
        )
    raise ValueError(f"Unsupported chunking strategy: {strategy}")
